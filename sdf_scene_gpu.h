
#pragma once
#include "LiteMath/LiteMath.h"
#include <vector>

using LiteMath::float2;
using LiteMath::float3;
using LiteMath::float4;
using LiteMath::uint2;
using LiteMath::uint3;
using LiteMath::uint4;
using LiteMath::int2;
using LiteMath::int3;
using LiteMath::int4;
using LiteMath::float4x4;
using LiteMath::float3x3;
using LiteMath::cross;
using LiteMath::dot;
using LiteMath::length;
using LiteMath::normalize;
using LiteMath::to_float4;
using LiteMath::to_float3;

// enum SdfPrimitiveType
static constexpr unsigned SDF_PRIM_SPHERE = 0;
static constexpr unsigned SDF_PRIM_BOX = 1;
static constexpr unsigned SDF_PRIM_CYLINDER = 2;
static constexpr unsigned SDF_PRIM_SIREN = 3;

struct SdfObject
{
  unsigned type;          // from enum SdfPrimitiveType
  unsigned params_offset; // in parameters vector
  unsigned params_count;
  unsigned neural_id = 0; // index in neural_properties if type is neural
  float distance_mult = 1.0f;
  float distance_add = 0.0f;
  float3 max_pos;
  float3 min_pos;
  float4x4 transform;
  unsigned complement = 0; // 0 or 1
};
struct SdfConjunction
{
  unsigned offset; // in objects vector
  unsigned size;
  float3 max_pos;
  float3 min_pos;
};

constexpr int NEURAL_SDF_MAX_LAYERS = 8;
constexpr int NEURAL_SDF_MAX_LAYER_SIZE = 1024;
constexpr float SIREN_W0 = 30;
struct NeuralProperties
{
  struct DenseLayer
  {
    unsigned offset;
    unsigned in_size;
    unsigned out_size;
  };

  unsigned layer_count;
  DenseLayer layers[NEURAL_SDF_MAX_LAYERS];
};

struct SdfHit
{
  unsigned hit_id = 0; // 0 if no hit
  float3 hit_pos;
  float3 hit_norm;
};

float2 box_intersects(const float3 &min_pos, const float3 &max_pos, const float3 &origin, const float3 &dir);

// evaluate distance to a specific primitive in scene
float eval_dist_prim(const float *parameters, const SdfObject *objects, const SdfConjunction *conjunctions, const NeuralProperties *neural_properties,
                     unsigned parameters_count, unsigned objects_count, unsigned conjunctions_count, unsigned neural_properties_count,
                     unsigned prim_id, float3 p);

// evaluate distance to a specific conjunction (a single primitive or intersection of primitives)
float eval_dist_conjunction(const float *parameters, const SdfObject *objects, const SdfConjunction *conjunctions, const NeuralProperties *neural_properties,
                            unsigned parameters_count, unsigned objects_count, unsigned conjunctions_count, unsigned neural_properties_count,
                            unsigned conj_id, float3 p);

// evaluate distance to a whole scene (minimum of distances to all conjunctions)
float eval_dist_scene(const float *parameters, const SdfObject *objects, const SdfConjunction *conjunctions, const NeuralProperties *neural_properties,
                      unsigned parameters_count, unsigned objects_count, unsigned conjunctions_count, unsigned neural_properties_count,
                      float3 p);

// perform sphere tracing to find ray intersection with a specific conjunction and inside given bbox
// (it can be smaller than real bbox of conjunction). Use with acceleration structure on conjunction bboxes
// dir vector MUST be normalized
SdfHit sdf_conjunction_sphere_tracing(const float *parameters, const SdfObject *objects, const SdfConjunction *conjunctions, const NeuralProperties *neural_properties,
                                      unsigned parameters_count, unsigned objects_count, unsigned conjunctions_count, unsigned neural_properties_count,
                                      unsigned conj_id, const float3 &min_pos, const float3 &max_pos,
                                      const float3 &pos, const float3 &dir, bool need_norm = false);