#include "sdf_scene.h"
#include <cassert>
#include <fstream>

using namespace LiteMath;

float tmp_mem[2*NEURAL_SDF_MAX_LAYER_SIZE];

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
  case SdfPrimitiveType::SIREN:
  {
    auto &prop = sdf.neural_properties[prim.neural_id];
    unsigned t_ofs1 = 0;
    unsigned t_ofs2 = NEURAL_SDF_MAX_LAYER_SIZE;

    tmp_mem[t_ofs1+0] = p.x;
    tmp_mem[t_ofs1+1] = p.y;
    tmp_mem[t_ofs1+2] = p.z;

    for (int l=0;l<prop.layer_count;l++)
    {
      unsigned m_ofs = prop.layers[l].offset;
      unsigned b_ofs = prop.layers[l].offset + prop.layers[l].in_size*prop.layers[l].out_size;
      for (int i=0;i<prop.layers[l].out_size;i++)
      {
        tmp_mem[t_ofs2 + i] = sdf.parameters[b_ofs + i];
        for (int j=0;j<prop.layers[l].in_size;j++)
          tmp_mem[t_ofs2 + i] += tmp_mem[t_ofs1 + j]*sdf.parameters[m_ofs + i*prop.layers[l].in_size + j];
        if (l < prop.layer_count-1)
          tmp_mem[t_ofs2 + i] = std::sin(SIREN_W0*tmp_mem[t_ofs2 + i]);      
      }

      t_ofs2 = t_ofs1;
      t_ofs1 = (t_ofs1 + NEURAL_SDF_MAX_LAYER_SIZE) % (2*NEURAL_SDF_MAX_LAYER_SIZE);
    }

    return tmp_mem[t_ofs1];
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

void load_neural_sdf_scene_SIREN(SdfScene &scene, const std::string &path)
{
  constexpr unsigned layers = 4;
  constexpr unsigned sz = 64;
  
  unsigned p_cnt = 3 * sz + 1 * sz + (layers - 1) * sz + 1 + (layers - 2) * sz * sz;
  scene.parameters.resize(p_cnt, 0.0f);
  std::ifstream fs(path, std::ios::binary);
  fs.read((char *)(scene.parameters.data()), sizeof(float) * p_cnt);

  scene.neural_properties.emplace_back();
  scene.neural_properties[0].layer_count = layers;
  scene.neural_properties[0].layers[0].in_size = 3;
  for (int i = 1; i < layers; i++)
    scene.neural_properties[0].layers[i].in_size = sz;
  for (int i = 0; i < layers - 1; i++)
    scene.neural_properties[0].layers[i].out_size = sz;
  scene.neural_properties[0].layers[layers - 1].out_size = 1;
  unsigned off = 0;
  for (int i = 0; i < layers; i++)
  {
    scene.neural_properties[0].layers[i].offset = off;
    off += (scene.neural_properties[0].layers[i].in_size + 1) * scene.neural_properties[0].layers[i].out_size;
  }

  scene.objects.emplace_back();
  scene.objects[0].type = SdfPrimitiveType::SIREN;
  scene.objects[0].params_offset = 0;
  scene.objects[0].params_count = p_cnt;
  scene.objects[0].bbox = AABB({-1, -1, -1}, {1, 1, 1});
  scene.objects[0].transform.identity();

  scene.conjunctions.emplace_back();
  scene.conjunctions[0].offset = 0;
  scene.conjunctions[0].size = 1;
  scene.conjunctions[0].bbox = scene.objects[0].bbox;
}

//Saves SdfScene as a separate hydra-xml file with no lights, materials and textures
//It's a lazy hack to do it without using Hydra API
//It saves both binary (<folder>/<name>.bin) and xml (<folder>/<name>.xml) files
void save_sdf_scene_hydra(const SdfScene &scene, const std::string &folder, const std::string &name)
{
  std::string bin_path = folder + "/" + name + ".bin";
  std::string path = folder + "/" + name + ".xml";
  save_sdf_scene(scene, bin_path);
  int bytesize = 3*sizeof(unsigned) + sizeof(SdfConjunction)*scene.conjunctions.size() + sizeof(SdfObject)*scene.objects.size() + 
                 sizeof(float)*scene.parameters.size();
  char buf[2<<12];
  snprintf(buf, 2<<12, R""""(
    <?xml version="1.0"?>
    <textures_lib>
    </textures_lib>
    <materials_lib>
    </materials_lib>
    <geometry_lib>
      <sdf id="0" name="sdf" type="bin" bytesize="%d" loc="%s">
      </sdf>
    </geometry_lib>
    <lights_lib>
    </lights_lib>
    <cam_lib>
      <camera id="0" name="my camera" type="uvn">
        <fov>60</fov>
        <nearClipPlane>0.01</nearClipPlane>
        <farClipPlane>100.0</farClipPlane>
        <up>0 1 0</up>
        <position>0 0 3</position>
        <look_at>0 0 0</look_at>
      </camera>
    </cam_lib>
    <render_lib>
      <render_settings type="HydraModern" id="0">
        <width>512</width>
        <height>512</height>
        <method_primary>pathtracing</method_primary>
        <method_secondary>pathtracing</method_secondary>
        <method_tertiary>pathtracing</method_tertiary>
        <method_caustic>pathtracing</method_caustic>
        <trace_depth>6</trace_depth>
        <diff_trace_depth>3</diff_trace_depth>
        <maxRaysPerPixel>1024</maxRaysPerPixel>
        <qmc_variant>7</qmc_variant>
      </render_settings>
    </render_lib>
    <scenes>
      <scene id="0" name="my scene" discard="1" bbox="-10 -10 -10 10 10 10">
        <instance id="0" mesh_id="0" rmap_id="0" scn_id="0" scn_sid="0" matrix="1 0 0 0   0 1 0 0   0 0 1 0   0 0 0 1 " />
        <instance id="1" mesh_id="1" rmap_id="0" scn_id="0" scn_sid="0" matrix="0.7 0 0 1    0 0.7 0 0   0 0 0.7 0   0 0 0 1 " />
      </scene>
    </scenes>
  )"""", bytesize, (name + ".bin").c_str());

  std::ofstream fs(path);
  fs << std::string(buf);
  fs.flush();
  fs.close();
}