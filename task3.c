#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <immintrin.h>
#include <stdint.h>

#define IMG_W       3840
#define IMG_H       2160
#define NUM_PIXELS  (IMG_W * IMG_H)
#define BUF_SIZE    (NUM_PIXELS * 3)
#define NUM_THREADS 4

static double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void generate_image(uint8_t *buf, size_t npixels) {
    for (size_t i = 0; i < npixels * 3; i++)
        buf[i] = rand() % 256;
}

static void write_ppm(const char *path, const uint8_t *buf, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror("fopen write"); exit(1); }
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    fwrite(buf, 1, w * h * 3, f);
    fclose(f);
}

static uint8_t *read_ppm(const char *path, int *w, int *h) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("fopen read"); exit(1); }
    int maxval;
    if (fscanf(f, "P6\n%d %d\n%d\n", w, h, &maxval) != 3) {
        fprintf(stderr, "Invalid PPM header\n"); exit(1);
    }
    size_t n = (size_t)(*w) * (*h) * 3;
    uint8_t *buf = malloc(n);
    if (!buf) { perror("malloc"); exit(1); }
    if (fread(buf, 1, n, f) != n) {
        fprintf(stderr, "Short read\n"); exit(1);
    }
    fclose(f);
    return buf;
}

/* ---- Scalar ---- */
static void scalar_gray(const uint8_t *in, uint8_t *out, size_t npixels) {
    for (size_t i = 0; i < npixels; i++) {
        uint8_t g = (uint8_t)(0.299f * in[i*3] + 0.587f * in[i*3+1] + 0.114f * in[i*3+2]);
        out[i*3]   = g;
        out[i*3+1] = g;
        out[i*3+2] = g;
    }
}

/* ---- SIMD ---- */
static void simd_gray(const uint8_t *in, uint8_t *out, size_t npixels) {
    const __m256 wr = _mm256_set1_ps(0.299f);
    const __m256 wg = _mm256_set1_ps(0.587f);
    const __m256 wb = _mm256_set1_ps(0.114f);

    float r8[8], g8[8], b8[8], gray8[8];
    size_t i = 0;

    for (; i + 8 <= npixels; i += 8) {
        for (int j = 0; j < 8; j++) {
            r8[j] = in[(i+j)*3 + 0];
            g8[j] = in[(i+j)*3 + 1];
            b8[j] = in[(i+j)*3 + 2];
        }
        __m256 vr   = _mm256_loadu_ps(r8);
        __m256 vg   = _mm256_loadu_ps(g8);
        __m256 vb   = _mm256_loadu_ps(b8);
        __m256 gray = _mm256_add_ps(
                        _mm256_add_ps(_mm256_mul_ps(vr, wr),
                                      _mm256_mul_ps(vg, wg)),
                        _mm256_mul_ps(vb, wb));
        _mm256_storeu_ps(gray8, gray);
        for (int j = 0; j < 8; j++) {
            uint8_t gv = (uint8_t)gray8[j];
            out[(i+j)*3]   = gv;
            out[(i+j)*3+1] = gv;
            out[(i+j)*3+2] = gv;
        }
    }
    for (; i < npixels; i++) {
        uint8_t gv = (uint8_t)(0.299f * in[i*3] + 0.587f * in[i*3+1] + 0.114f * in[i*3+2]);
        out[i*3] = out[i*3+1] = out[i*3+2] = gv;
    }
}

/* ---- Multithreading (scalar per thread) ---- */
typedef struct {
    const uint8_t *in;
    uint8_t       *out;
    size_t         start;
    size_t         count;
} ThreadArg;

static void *mt_worker(void *arg) {
    ThreadArg *a = (ThreadArg *)arg;
    scalar_gray(a->in + a->start * 3, a->out + a->start * 3, a->count);
    return NULL;
}

static void mt_gray(const uint8_t *in, uint8_t *out, size_t npixels) {
    pthread_t  threads[NUM_THREADS];
    ThreadArg  args[NUM_THREADS];
    size_t     chunk = npixels / NUM_THREADS;
    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].in    = in;
        args[i].out   = out;
        args[i].start = i * chunk;
        args[i].count = (i == NUM_THREADS - 1) ? (npixels - i * chunk) : chunk;
        pthread_create(&threads[i], NULL, mt_worker, &args[i]);
    }
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);
}

/* ---- SIMD + Multithreading ---- */
static void *simd_mt_worker(void *arg) {
    ThreadArg *a = (ThreadArg *)arg;
    simd_gray(a->in + a->start * 3, a->out + a->start * 3, a->count);
    return NULL;
}

static void simd_mt_gray(const uint8_t *in, uint8_t *out, size_t npixels) {
    pthread_t  threads[NUM_THREADS];
    ThreadArg  args[NUM_THREADS];
    size_t     chunk = npixels / NUM_THREADS;
    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].in    = in;
        args[i].out   = out;
        args[i].start = i * chunk;
        args[i].count = (i == NUM_THREADS - 1) ? (npixels - i * chunk) : chunk;
        pthread_create(&threads[i], NULL, simd_mt_worker, &args[i]);
    }
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);
}

static int verify(const uint8_t *ref, const uint8_t *cmp, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (ref[i] != cmp[i]) return 0;
    return 1;
}

int main(void) {
    srand(42);

    uint8_t *src        = malloc(BUF_SIZE);
    uint8_t *out_scalar = malloc(BUF_SIZE);
    uint8_t *out_simd   = malloc(BUF_SIZE);
    uint8_t *out_mt     = malloc(BUF_SIZE);
    uint8_t *out_both   = malloc(BUF_SIZE);
    if (!src || !out_scalar || !out_simd || !out_mt || !out_both) {
        perror("malloc"); return 1;
    }

    printf("Generating image %dx%d...\n", IMG_W, IMG_H);
    generate_image(src, NUM_PIXELS);
    write_ppm("input.ppm", src, IMG_W, IMG_H);

    /* read back (demonstrates PPM reading) */
    int rw, rh;
    uint8_t *loaded = read_ppm("input.ppm", &rw, &rh);
    memcpy(src, loaded, BUF_SIZE);
    free(loaded);

    printf("Image size: %d x %d\n", IMG_W, IMG_H);
    printf("Threads used: %d\n\n", NUM_THREADS);

    double t0, t1;

    t0 = get_time();
    scalar_gray(src, out_scalar, NUM_PIXELS);
    t1 = get_time();
    printf("Scalar time:               %.3f sec\n", t1 - t0);

    t0 = get_time();
    simd_gray(src, out_simd, NUM_PIXELS);
    t1 = get_time();
    printf("SIMD time:                 %.3f sec\n", t1 - t0);

    t0 = get_time();
    mt_gray(src, out_mt, NUM_PIXELS);
    t1 = get_time();
    printf("Multithreading time:       %.3f sec\n", t1 - t0);

    t0 = get_time();
    simd_mt_gray(src, out_both, NUM_PIXELS);
    t1 = get_time();
    printf("Multithreading + SIMD time:%.3f sec\n", t1 - t0);

    int ok = verify(out_scalar, out_simd, BUF_SIZE) &&
             verify(out_scalar, out_mt,   BUF_SIZE) &&
             verify(out_scalar, out_both, BUF_SIZE);
    printf("\nVerification: %s\n", ok ? "PASSED" : "FAILED");

    write_ppm("gray_output.ppm", out_scalar, IMG_W, IMG_H);
    printf("Output image: gray_output.ppm\n");

    free(src); free(out_scalar); free(out_simd); free(out_mt); free(out_both);
    return 0;
}