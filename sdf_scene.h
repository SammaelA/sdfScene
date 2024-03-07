#pragma once
#include "LiteMath_ext.h"
#include <string>
#include <vector>

enum class SdfPrimitiveType
{
  SPHERE,
  BOX,
  CYLINDER
};

struct SdfObject
{
  SdfPrimitiveType type;
  unsigned params_offset; // in parameters vector
  unsigned params_count;
  float distance_mult = 1.0f;
  float distance_add = 0.0f;
  LiteMath::AABB bbox;
  LiteMath::float4x4 transform;
  bool complement = false;
};
struct SdfConjunction
{
  unsigned offset; // in objects vector
  unsigned size;
  LiteMath::AABB bbox;
};
struct SdfScene
{
  std::vector<float> parameters;
  std::vector<SdfObject> objects;
  std::vector<SdfConjunction> conjunctions;
};

// evaluate distance to a specific primitive in scene
float eval_dist_prim(const SdfScene &sdf, unsigned prim_id, LiteMath::float3 p);

// evaluate distance to a specific conjunction (a single primitive or intersection of primitives)
float eval_dist_conjunction(const SdfScene &sdf, unsigned conj_id, LiteMath::float3 p);

// evaluate distance to a whole scene (minimum of distances to all conjunctions)
float eval_dist_scene(const SdfScene &sdf, LiteMath::float3 p);

// perform sphere tracing to find ray intersection with a specific conjunction and inside given bbox
// (it can be smaller than real bbox of conjunction). Use with acceleration structure on conjunction bboxes
// dir vector MUST be normalized
bool sdf_conjunction_sphere_tracing(const SdfScene &sdf, unsigned conj_id, const LiteMath::AABB &bbox, 
                                    const LiteMath::float3 &pos, const LiteMath::float3 &dir,
                                    LiteMath::float3 *surface_pos = nullptr,
                                    LiteMath::float3 *surface_normal = nullptr);

// perform sphere tracing to find ray intersection with a whole scene and inside given bbox
// dir vector MUST be normalized
bool sdf_sphere_tracing(const SdfScene &sdf, const LiteMath::AABB &sdf_bbox, const LiteMath::float3 &pos, const LiteMath::float3 &dir,
                        LiteMath::float3 *surface_pos = nullptr);

// save/load scene
void save_sdf_scene(const SdfScene &scene, const std::string &path);
void load_sdf_scene(SdfScene &scene, const std::string &path);