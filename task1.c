#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <immintrin.h>

#define DNA_SIZE (256 * 1024 * 1024)
#define NUM_THREADS 4
#define NUCLEOTIDES "ACGT"

static char *dna_buffer;

typedef struct {
    long a, c, g, t;
} Counts;

static Counts global_counts;
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    char *start;
    size_t len;
    Counts local;
} ThreadArg;

static double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void generate_dna(char *buf, size_t n) {
    const char bases[4] = {'A', 'C', 'G', 'T'};
    for (size_t i = 0; i < n; i++)
        buf[i] = bases[rand() % 4];
}

static Counts scalar(const char *buf, size_t n) {
    Counts c = {0};
    for (size_t i = 0; i < n; i++) {
        switch (buf[i]) {
            case 'A': c.a++; break;
            case 'C': c.c++; break;
            case 'G': c.g++; break;
            case 'T': c.t++; break;
        }
    }
    return c;
}

static void *mt_worker(void *arg) {
    ThreadArg *a = (ThreadArg *)arg;
    Counts local = {0};
    for (size_t i = 0; i < a->len; i++) {
        switch (a->start[i]) {
            case 'A': local.a++; break;
            case 'C': local.c++; break;
            case 'G': local.g++; break;
            case 'T': local.t++; break;
        }
    }
    pthread_mutex_lock(&mutex);
    global_counts.a += local.a;
    global_counts.c += local.c;
    global_counts.g += local.g;
    global_counts.t += local.t;
    pthread_mutex_unlock(&mutex);
    return NULL;
}

static Counts multithreading(const char *buf, size_t n) {
    pthread_t threads[NUM_THREADS];
    ThreadArg args[NUM_THREADS];
    global_counts = (Counts){0};
    size_t chunk = n / NUM_THREADS;
    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].start = (char *)buf + i * chunk;
        args[i].len = (i == NUM_THREADS - 1) ? (n - i * chunk) : chunk;
        pthread_create(&threads[i], NULL, mt_worker, &args[i]);
    }
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);
    return global_counts;
}

static Counts simd_count(const char *buf, size_t n) {
    __m256i va = _mm256_set1_epi8('A');
    __m256i vc = _mm256_set1_epi8('C');
    __m256i vg = _mm256_set1_epi8('G');
    __m256i vt = _mm256_set1_epi8('T');

    long a = 0, c = 0, g = 0, t = 0;
    size_t i = 0;
    size_t limit = n - (n % 32);

    for (; i < limit; i += 32) {
        __m256i chunk = _mm256_loadu_si256((__m256i *)(buf + i));
        __m256i ma = _mm256_cmpeq_epi8(chunk, va);
        __m256i mc = _mm256_cmpeq_epi8(chunk, vc);
        __m256i mg = _mm256_cmpeq_epi8(chunk, vg);
        __m256i mt = _mm256_cmpeq_epi8(chunk, vt);
        a -= (long)(int)_mm256_movemask_epi8(ma);
        c -= (long)(int)_mm256_movemask_epi8(mc);
        g -= (long)(int)_mm256_movemask_epi8(mg);
        t -= (long)(int)_mm256_movemask_epi8(mt);
    }

    /* Use popcount to count set bits from movemask */
    /* Recount using __builtin_popcount approach */
    /* Reset and redo properly */
    a = 0; c = 0; g = 0; t = 0;
    i = 0;
    for (; i < limit; i += 32) {
        __m256i chunk = _mm256_loadu_si256((__m256i *)(buf + i));
        a += __builtin_popcount((unsigned int)_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, va)));
        c += __builtin_popcount((unsigned int)_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, vc)));
        g += __builtin_popcount((unsigned int)_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, vg)));
        t += __builtin_popcount((unsigned int)_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, vt)));
    }

    for (; i < n; i++) {
        switch (buf[i]) {
            case 'A': a++; break;
            case 'C': c++; break;
            case 'G': g++; break;
            case 'T': t++; break;
        }
    }

    return (Counts){a, c, g, t};
}

static void *simd_mt_worker(void *arg) {
    ThreadArg *a = (ThreadArg *)arg;
    a->local = simd_count(a->start, a->len);
    return NULL;
}

static Counts simd_multithreading(const char *buf, size_t n) {
    pthread_t threads[NUM_THREADS];
    ThreadArg args[NUM_THREADS];
    size_t chunk = n / NUM_THREADS;
    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].start = (char *)buf + i * chunk;
        args[i].len = (i == NUM_THREADS - 1) ? (n - i * chunk) : chunk;
        pthread_create(&threads[i], NULL, simd_mt_worker, &args[i]);
    }
    Counts total = {0};
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
        total.a += args[i].local.a;
        total.c += args[i].local.c;
        total.g += args[i].local.g;
        total.t += args[i].local.t;
    }
    return total;
}

static void print_counts(const char *label, Counts cnt, double elapsed) {
    printf("%-28s A=%-10ld C=%-10ld G=%-10ld T=%-10ld  time: %.3f sec\n",
           label, cnt.a, cnt.c, cnt.g, cnt.t, elapsed);
}

int main(void) {
    srand(42);
    dna_buffer = (char *)malloc(DNA_SIZE);
    if (!dna_buffer) { perror("malloc"); return 1; }
    generate_dna(dna_buffer, DNA_SIZE);

    printf("DNA size: %d MB\n", DNA_SIZE / (1024 * 1024));
    printf("Threads used: %d\n\n", NUM_THREADS);

    double t0, t1;
    Counts r;

    t0 = get_time();
    r = scalar(dna_buffer, DNA_SIZE);
    t1 = get_time();
    print_counts("Scalar:", r, t1 - t0);

    t0 = get_time();
    r = multithreading(dna_buffer, DNA_SIZE);
    t1 = get_time();
    print_counts("Multithreading:", r, t1 - t0);

    t0 = get_time();
    r = simd_count(dna_buffer, DNA_SIZE);
    t1 = get_time();
    print_counts("SIMD:", r, t1 - t0);

    t0 = get_time();
    r = simd_multithreading(dna_buffer, DNA_SIZE);
    t1 = get_time();
    print_counts("SIMD + Multithreading:", r, t1 - t0);

    free(dna_buffer);
    return 0;
}