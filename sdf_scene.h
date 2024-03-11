#pragma once
#include "LiteMath_ext.h"
#include <string>
#include <vector>

enum class SdfPrimitiveType
{
  SPHERE,
  BOX,
  CYLINDER,
  SIREN
};

struct SdfObject
{
  SdfPrimitiveType type;
  unsigned params_offset; // in parameters vector
  unsigned params_count;
  unsigned neural_id = 0; // index in neural_properties if type is neural 
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

struct SdfScene
{
  std::vector<float> parameters;
  std::vector<SdfObject> objects;
  std::vector<SdfConjunction> conjunctions;
  std::vector<NeuralProperties> neural_properties;
};

//all interfaces use SdfSceneView to be independant of how exactly 
//SDF scenes are stored
struct SdfSceneView
{
  SdfSceneView() = default;
  SdfSceneView(const SdfScene &scene)
  {
    parameters = scene.parameters.data();
    objects = scene.objects.data();
    conjunctions = scene.conjunctions.data();
    neural_properties = scene.neural_properties.data();

    parameters_count = scene.parameters.size();
    objects_count = scene.objects.size();
    conjunctions_count = scene.conjunctions.size();
    neural_properties_count = scene.neural_properties.size();
  }
  SdfSceneView(const std::vector<float> &_parameters, 
               const std::vector<SdfObject> &_objects,
               const std::vector<SdfConjunction> &_conjunctions,
               const std::vector<NeuralProperties> &_neural_properties)
  {
    parameters = _parameters.data();
    objects = _objects.data();
    conjunctions = _conjunctions.data();
    neural_properties = _neural_properties.data();

    parameters_count = _parameters.size();
    objects_count = _objects.size();
    conjunctions_count = _conjunctions.size();
    neural_properties_count = _neural_properties.size();
  }

  const float *parameters;
  const SdfObject *objects;
  const SdfConjunction *conjunctions;
  const NeuralProperties *neural_properties;

  unsigned parameters_count;
  unsigned objects_count;
  unsigned conjunctions_count;
  unsigned neural_properties_count;
};

// evaluate distance to a specific primitive in scene
float eval_dist_prim(const SdfSceneView &sdf, unsigned prim_id, LiteMath::float3 p);

// evaluate distance to a specific conjunction (a single primitive or intersection of primitives)
float eval_dist_conjunction(const SdfSceneView &sdf, unsigned conj_id, LiteMath::float3 p);

// evaluate distance to a whole scene (minimum of distances to all conjunctions)
float eval_dist_scene(const SdfSceneView &sdf, LiteMath::float3 p);

// perform sphere tracing to find ray intersection with a specific conjunction and inside given bbox
// (it can be smaller than real bbox of conjunction). Use with acceleration structure on conjunction bboxes
// dir vector MUST be normalized
bool sdf_conjunction_sphere_tracing(const SdfSceneView &sdf, unsigned conj_id, const LiteMath::AABB &bbox, 
                                    const LiteMath::float3 &pos, const LiteMath::float3 &dir,
                                    LiteMath::float3 *surface_pos = nullptr,
                                    LiteMath::float3 *surface_normal = nullptr);

// perform sphere tracing to find ray intersection with a whole scene and inside given bbox
// dir vector MUST be normalized
bool sdf_sphere_tracing(const SdfSceneView &sdf, const LiteMath::AABB &sdf_bbox, const LiteMath::float3 &pos, const LiteMath::float3 &dir,
                        LiteMath::float3 *surface_pos = nullptr);

// save/load scene
void save_sdf_scene_hydra(const SdfScene &scene, const std::string &folder, const std::string &name);
void save_sdf_scene(const SdfScene &scene, const std::string &path);
void load_sdf_scene(SdfScene &scene, const std::string &path);
void load_neural_sdf_scene_SIREN(SdfScene &scene, const std::string &path); //loads scene from raw SIREN weights file