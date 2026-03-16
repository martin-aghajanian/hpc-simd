#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <immintrin.h>

#define BUF_SIZE  (256 * 1024 * 1024)
#define NUM_THREADS 4

static double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void fill_buffer(char *buf, size_t n) {
    const char charset[] =
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789 !@#$%^&*()-_=+";
    size_t len = sizeof(charset) - 1;
    for (size_t i = 0; i < n; i++)
        buf[i] = charset[rand() % len];
}

static void to_upper_scalar(char *buf, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (buf[i] >= 'a' && buf[i] <= 'z')
            buf[i] -= 32;
}

typedef struct {
    char *start;
    size_t len;
} ThreadArg;

static void *mt_worker(void *arg) {
    ThreadArg *a = (ThreadArg *)arg;
    to_upper_scalar(a->start, a->len);
    return NULL;
}

static void multithreading(char *buf, size_t n) {
    pthread_t threads[NUM_THREADS];
    ThreadArg args[NUM_THREADS];
    size_t chunk = n / NUM_THREADS;
    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].start = buf + i * chunk;
        args[i].len = (i == NUM_THREADS - 1) ? (n - i * chunk) : chunk;
        pthread_create(&threads[i], NULL, mt_worker, &args[i]);
    }
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);
}

static void simd_upper(char *buf, size_t n) {
    __m256i lower_a  = _mm256_set1_epi8('a');
    __m256i lower_z  = _mm256_set1_epi8('z');
    __m256i delta    = _mm256_set1_epi8(32);

    size_t i = 0;
    size_t limit = n - (n % 32);

    for (; i < limit; i += 32) {
        __m256i v    = _mm256_loadu_si256((__m256i *)(buf + i));
        __m256i ge_a = _mm256_cmpgt_epi8(v, _mm256_sub_epi8(lower_a, _mm256_set1_epi8(1)));
        __m256i le_z = _mm256_cmpgt_epi8(_mm256_add_epi8(lower_z, _mm256_set1_epi8(1)), v);
        __m256i mask = _mm256_and_si256(ge_a, le_z);
        __m256i sub  = _mm256_and_si256(mask, delta);
        __m256i res  = _mm256_sub_epi8(v, sub);
        _mm256_storeu_si256((__m256i *)(buf + i), res);
    }

    for (; i < n; i++)
        if (buf[i] >= 'a' && buf[i] <= 'z')
            buf[i] -= 32;
}

static void *simd_mt_worker(void *arg) {
    ThreadArg *a = (ThreadArg *)arg;
    simd_upper(a->start, a->len);
    return NULL;
}

static void simd_multithreading(char *buf, size_t n) {
    pthread_t threads[NUM_THREADS];
    ThreadArg args[NUM_THREADS];
    size_t chunk = n / NUM_THREADS;
    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].start = buf + i * chunk;
        args[i].len = (i == NUM_THREADS - 1) ? (n - i * chunk) : chunk;
        pthread_create(&threads[i], NULL, simd_mt_worker, &args[i]);
    }
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);
}

static int verify(const char *a, const char *b, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (a[i] != b[i]) return 0;
    return 1;
}

int main(void) {
    srand(42);

    char *original = malloc(BUF_SIZE);
    char *buf_mt   = malloc(BUF_SIZE);
    char *buf_simd = malloc(BUF_SIZE);
    char *buf_both = malloc(BUF_SIZE);

    if (!original || !buf_mt || !buf_simd || !buf_both) {
        perror("malloc");
        return 1;
    }

    fill_buffer(original, BUF_SIZE);
    memcpy(buf_mt,   original, BUF_SIZE);
    memcpy(buf_simd, original, BUF_SIZE);
    memcpy(buf_both, original, BUF_SIZE);

    printf("Buffer size: %d MB\n", BUF_SIZE / (1024 * 1024));
    printf("Threads used: %d\n\n", NUM_THREADS);

    double t0, t1;

    t0 = get_time();
    multithreading(buf_mt, BUF_SIZE);
    t1 = get_time();
    printf("Multithreading time:      %.3f sec\n", t1 - t0);

    t0 = get_time();
    simd_upper(buf_simd, BUF_SIZE);
    t1 = get_time();
    printf("SIMD time:                %.3f sec\n", t1 - t0);

    t0 = get_time();
    simd_multithreading(buf_both, BUF_SIZE);
    t1 = get_time();
    printf("SIMD + Multithreading:    %.3f sec\n", t1 - t0);

    printf("\nVerification (MT vs SIMD):      %s\n", verify(buf_mt, buf_simd, BUF_SIZE) ? "PASS" : "FAIL");
    printf("Verification (MT vs SIMD+MT):   %s\n", verify(buf_mt, buf_both, BUF_SIZE) ? "PASS" : "FAIL");

    free(original);
    free(buf_mt);
    free(buf_simd);
    free(buf_both);
    return 0;
}