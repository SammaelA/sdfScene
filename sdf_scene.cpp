#include "sdf_scene.h"
#include <cassert>
#include <fstream>

using namespace LiteMath;

float eval_dist_prim(const SdfSceneView &sdf, unsigned prim_id, float3 p)
{
  const SdfObject &prim = sdf.objects[prim_id];
  float3 pos = prim.transform * p;
  // printf("%f %f %f -- %f %f %f\n", p.x, p.y, p.z, pos.x, pos.y, pos.z);
  // printf("%f %f %f %f\n %f %f %f %f\n %f %f %f %f\n %f %f %f %f\n",
  //        prim.transform(0, 0), prim.transform(0, 1), prim.transform(0, 2), prim.transform(0, 3),
  //        prim.transform(1, 0), prim.transform(1, 1), prim.transform(1, 2), prim.transform(1, 3),
  //        prim.transform(2, 0), prim.transform(2, 1), prim.transform(2, 2), prim.transform(2, 3),
  //        prim.transform(3, 0), prim.transform(3, 1), prim.transform(3, 2), prim.transform(3, 3));

  switch (prim.type)
  {
  case SdfPrimitiveType::SPHERE:
  {
    float r = sdf.parameters[prim.params_offset + 0];
    // fprintf(stderr, "sphere %f %f %f - %f",pos.x, pos.y, pos.z, r);
    return length(pos) - r;
  }
  case SdfPrimitiveType::BOX:
  {
    float3 size(sdf.parameters[prim.params_offset + 0],
                sdf.parameters[prim.params_offset + 1],
                sdf.parameters[prim.params_offset + 2]);
    // fprintf(stderr, "box %f %f %f - %f %f %f - %f %f %f",p.x, p.y, p.z, pos.x, pos.y, pos.z, size.x, size.y, size.z);
    float3 q = abs(pos) - size;
    return length(max(q, float3(0.0f))) + min(max(q.x, max(q.y, q.z)), 0.0f);
  }
  case SdfPrimitiveType::CYLINDER:
  {
    float h = sdf.parameters[prim.params_offset + 0];
    float r = sdf.parameters[prim.params_offset + 1];
    float2 d = abs(float2(sqrt(pos.x * pos.x + pos.z * pos.z), pos.y)) - float2(r, h);
    return min(max(d.x, d.y), 0.0f) + length(max(d, float2(0.0f)));
  }
  default:
    fprintf(stderr, "unknown type %u", prim.type);
    assert(false);
    break;
  }
  return -1000;
}

float eval_dist_conjunction(const SdfSceneView &sdf, unsigned conj_id, LiteMath::float3 p)
{
  const SdfConjunction &conj = sdf.conjunctions[conj_id];
  float conj_d = -1e6;
  for (unsigned pid = conj.offset; pid < conj.offset + conj.size; pid++)
  {
    float prim_d = sdf.objects[pid].distance_mult * eval_dist_prim(sdf, pid, p) + sdf.objects[pid].distance_add;
    conj_d = max(conj_d, sdf.objects[pid].complement ? -prim_d : prim_d);
  }
  return conj_d;
}

float eval_dist_scene(const SdfSceneView &sdf, float3 p)
{
  float d = 1e6;
  for (unsigned i=0;i<sdf.conjunctions_count;i++)
    d = min(d, eval_dist_conjunction(sdf, i, p));
  return d;
}

bool sdf_conjunction_sphere_tracing(const SdfSceneView &sdf, unsigned conj_id, const LiteMath::AABB &sdf_bbox, 
                                    const LiteMath::float3 &pos, const LiteMath::float3 &dir,
                                    LiteMath::float3 *surface_pos,
                                    LiteMath::float3 *surface_normal)
{
  constexpr float EPS = 1e-5;
  float t = 0;
  float tFar = 1e4;
  if (!sdf_bbox.contains(pos))
  {
    if (!sdf_bbox.intersects(pos, dir, &t, &tFar))
      return false;
  }
  int iter = 0;
  float d = eval_dist_conjunction(sdf, conj_id, pos + t * dir);
  while (iter < 1000 && d > EPS && t < tFar)
  {
    t += d + EPS;
    d = eval_dist_conjunction(sdf, conj_id, pos + t * dir);
    iter++;
  }

  if (surface_pos)
    *surface_pos = pos + t * dir;

  if (surface_normal)
  {
    float3 p0 = pos + t * dir;
    constexpr float h = 0.001;
    float ddx = (eval_dist_conjunction(sdf, conj_id, p0 + LiteMath::float3(h, 0, 0)) - eval_dist_conjunction(sdf, conj_id, p0 + LiteMath::float3(-h, 0, 0))) / (2 * h);
    float ddy = (eval_dist_conjunction(sdf, conj_id, p0 + LiteMath::float3(0, h, 0)) - eval_dist_conjunction(sdf, conj_id, p0 + LiteMath::float3(0, -h, 0))) / (2 * h);
    float ddz = (eval_dist_conjunction(sdf, conj_id, p0 + LiteMath::float3(0, 0, h)) - eval_dist_conjunction(sdf, conj_id, p0 + LiteMath::float3(0, 0, -h))) / (2 * h);
    
    *surface_normal = normalize(float3(ddx, ddy, ddz));
    //fprintf(stderr, "st %d (%f %f %f)\n", iter, surface_normal->x, surface_normal->y, surface_normal->z);
  }
  // fprintf(stderr, "st %d (%f %f %f)", iter, p0.x, p0.y, p0.z);
  return d <= EPS;
}

bool sdf_sphere_tracing(const SdfSceneView &sdf, const AABB &sdf_bbox, const float3 &pos, const float3 &dir,
                        float3 *surface_pos)
{
  constexpr float EPS = 1e-5;
  float t = 0;
  float tFar = 1e4;
  if (!sdf_bbox.contains(pos))
  {
    if (!sdf_bbox.intersects(pos, dir, &t, &tFar))
      return false;
  }
  int iter = 0;
  float d = eval_dist_scene(sdf, pos + t * dir);
  while (iter < 1000 && d > EPS && t < tFar)
  {
    t += d + EPS;
    d = eval_dist_scene(sdf, pos + t * dir);
    iter++;
  }
  if (surface_pos)
    *surface_pos = pos + t * dir;

  return d <= EPS;
}

void save_sdf_scene(const SdfScene &scene, const std::string &path)
{
  std::ofstream fs(path, std::ios::binary);
  unsigned c_count = scene.conjunctions.size();
  unsigned o_count = scene.objects.size();
  unsigned p_count = scene.parameters.size();

  fs.write((const char *)(&c_count), sizeof(unsigned));
  fs.write((const char *)(&o_count), sizeof(unsigned));
  fs.write((const char *)(&p_count), sizeof(unsigned));

  fs.write((const char *)scene.conjunctions.data(), c_count * sizeof(SdfConjunction));
  fs.write((const char *)scene.objects.data(), o_count * sizeof(SdfObject));
  fs.write((const char *)scene.parameters.data(), p_count * sizeof(float));
  fs.flush();
  fs.close();
}

void load_sdf_scene(SdfScene &scene, const std::string &path)
{
  std::ifstream fs(path, std::ios::binary);
  unsigned c_count = 0;
  unsigned o_count = 0;
  unsigned p_count = 0;

  fs.read((char *)(&c_count), sizeof(unsigned));
  fs.read((char *)(&o_count), sizeof(unsigned));
  fs.read((char *)(&p_count), sizeof(unsigned));

  assert(c_count > 0);
  assert(o_count > 0);
  assert(p_count > 0);
  scene.conjunctions.resize(c_count);
  scene.objects.resize(o_count);
  scene.parameters.resize(p_count);

  fs.read((char *)scene.conjunctions.data(), c_count * sizeof(SdfConjunction));
  fs.read((char *)scene.objects.data(), o_count * sizeof(SdfObject));
  fs.read((char *)scene.parameters.data(), p_count * sizeof(float));
  fs.close();
}