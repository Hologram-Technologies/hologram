#ifndef HOLOGRAM_H
#define HOLOGRAM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HOLOGRAM_ABI_VERSION 1u
#define HOLOGRAM_DTYPE_DEFAULT 0u
#define HOLOGRAM_DTYPE_F32 8u
#define HOLOGRAM_ERROR_NONE 0
#define HOLOGRAM_ERROR_PARSE 1
#define HOLOGRAM_ERROR_GRAPH 2
#define HOLOGRAM_ERROR_UNSUPPORTED_OP 3
#define HOLOGRAM_ERROR_BAD_ATTR 4
#define HOLOGRAM_ERROR_SHAPE 5
#define HOLOGRAM_ERROR_EXTERNAL_TENSOR 6
#define HOLOGRAM_ERROR_ARCHIVE_LOAD 7
#define HOLOGRAM_ERROR_EXECUTION 8
#define HOLOGRAM_ERROR_ABI_MISMATCH 9
#define HOLOGRAM_ERROR_INVALID_ARGUMENT 10
#define HOLOGRAM_ERROR_UNSUPPORTED_DTYPE 11
#define HOLOGRAM_ERROR_COMPILE 12

typedef struct HologramSourceBuilder HologramSourceBuilder;

typedef struct HologramString {
  const uint8_t *ptr;
  size_t len;
} HologramString;

typedef struct HologramShape {
  const uint64_t *dims;
  size_t rank;
} HologramShape;

typedef struct HologramTensorDesc {
  HologramString name;
  uint8_t dtype_id;
  HologramShape shape;
} HologramTensorDesc;

typedef struct HologramConstDesc {
  HologramTensorDesc tensor;
  const uint8_t *bytes;
  size_t byte_len;
} HologramConstDesc;

typedef struct HologramExternalTensorDesc {
  HologramTensorDesc tensor;
  HologramString path;
  uint64_t byte_offset;
  uint64_t byte_len;
  uint8_t content_hash[32];
} HologramExternalTensorDesc;

typedef struct HologramSourceOp {
  HologramString output;
  HologramString op;
  const HologramString *inputs;
  size_t input_count;
  HologramShape shape;
} HologramSourceOp;

uint32_t hologram_abi_version(void);
uint32_t hologram_archive_format_version(void);
int32_t hologram_feature_supported(HologramString feature);
int32_t hologram_last_error(void);
int32_t hologram_last_error_code(void);
const char *hologram_error_message(void);
const char *hologram_last_error_message(void);
size_t hologram_last_error_line(void);
size_t hologram_last_error_column(void);
const char *hologram_last_error_rejected(void);

HologramSourceBuilder *hologram_source_builder_new(void);
void hologram_source_builder_free(HologramSourceBuilder *builder);
int32_t hologram_source_builder_input(HologramSourceBuilder *builder,
                                      const HologramTensorDesc *desc);
int32_t hologram_source_builder_const(HologramSourceBuilder *builder,
                                      const HologramConstDesc *desc);
int32_t hologram_source_builder_const_ref(HologramSourceBuilder *builder,
                                          const HologramExternalTensorDesc *desc);
int32_t hologram_source_builder_op(HologramSourceBuilder *builder,
                                   const HologramSourceOp *op);
int32_t hologram_source_builder_output(HologramSourceBuilder *builder,
                                       HologramString name);
int32_t hologram_source_builder_output_alias(HologramSourceBuilder *builder,
                                             HologramString name,
                                             HologramString source);
int32_t hologram_source_builder_compile(const HologramSourceBuilder *builder,
                                        uint8_t *out,
                                        size_t out_capacity);

int32_t hologram_compile_empty(uint8_t *out, size_t out_capacity);
int32_t hologram_compile_source(const uint8_t *source_ptr,
                                size_t source_len,
                                uint8_t *out,
                                size_t out_capacity);

int32_t hologram_session_load(const uint8_t *archive_ptr, size_t archive_len);
int32_t hologram_session_input_count(int32_t handle);
int32_t hologram_session_output_count(int32_t handle);
int32_t hologram_session_kernel_count(int32_t handle);
int32_t hologram_session_output_byte_len(int32_t handle, size_t i);
int32_t hologram_session_input_dtype(int32_t handle, size_t i);
int32_t hologram_session_output_dtype(int32_t handle, size_t i);
int32_t hologram_session_archive_fingerprint(int32_t handle, uint8_t *out);
int32_t hologram_session_execute(int32_t handle,
                                 const uint8_t *const *in_ptrs,
                                 const size_t *in_lens,
                                 size_t in_count,
                                 uint8_t *const *out_ptrs,
                                 const size_t *out_caps,
                                 size_t out_count);
int32_t hologram_session_close(int32_t handle);
int32_t hologram_session_input_name(int32_t handle,
                                    size_t i,
                                    uint8_t *out,
                                    size_t out_capacity);
int32_t hologram_session_output_name(int32_t handle,
                                     size_t i,
                                     uint8_t *out,
                                     size_t out_capacity);
int32_t hologram_session_input_shape(int32_t handle,
                                     size_t i,
                                     uint64_t *out_dims,
                                     size_t max_dims);
int32_t hologram_session_output_shape(int32_t handle,
                                      size_t i,
                                      uint64_t *out_dims,
                                      size_t max_dims);
int32_t hologram_session_extension(int32_t handle,
                                   const uint8_t *key_ptr,
                                   size_t key_len,
                                   uint8_t *out,
                                   size_t out_capacity);

#ifdef __cplusplus
}
#endif

#endif
