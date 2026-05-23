#include "liftoff_render_api.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

struct LiftoffRenderMeshVertex {
    double x;
    double y;
    double z;
};

struct LiftoffRenderMeshTriangle {
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint8_t r;
    uint8_t g;
    uint8_t b8;
};

struct LiftoffRenderEngine {
    LiftoffRenderConfig config;
    std::vector<uint8_t> pixels;
    bool target_mesh_load_attempted = false;
    bool target_mesh_loaded = false;
    std::vector<LiftoffRenderMeshVertex> target_mesh_vertices;
    std::vector<LiftoffRenderMeshTriangle> target_mesh_triangles;
};

namespace {

struct Vec3d {
    double x;
    double y;
    double z;
};

Vec3d operator+(Vec3d a, Vec3d b) {
    return Vec3d{a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3d operator-(Vec3d a, Vec3d b) {
    return Vec3d{a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3d operator*(double s, Vec3d v) {
    return Vec3d{s * v.x, s * v.y, s * v.z};
}

double dot(Vec3d a, Vec3d b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3d cross(Vec3d a, Vec3d b) {
    return Vec3d{
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

double norm(Vec3d v) {
    return std::sqrt(dot(v, v));
}

Vec3d normalize(Vec3d v) {
    double n = norm(v);
    if (n <= 1e-12) {
        return Vec3d{1.0, 0.0, 0.0};
    }
    return (1.0 / n) * v;
}

Vec3d as_vec3(const LiftoffRenderVec3& v) {
    return Vec3d{v.x, v.y, v.z};
}

Vec3d quat_rotate(const LiftoffRenderQuatXyzw& q, Vec3d v) {
    Vec3d qv{q.x, q.y, q.z};
    Vec3d t{
        2.0 * (qv.y * v.z - qv.z * v.y),
        2.0 * (qv.z * v.x - qv.x * v.z),
        2.0 * (qv.x * v.y - qv.y * v.x),
    };
    Vec3d cross_q_t{
        qv.y * t.z - qv.z * t.y,
        qv.z * t.x - qv.x * t.z,
        qv.x * t.y - qv.y * t.x,
    };
    return v + q.w * t + cross_q_t;
}

Vec3d quat_inverse_rotate(const LiftoffRenderQuatXyzw& q, Vec3d v) {
    LiftoffRenderQuatXyzw inv{-q.x, -q.y, -q.z, q.w};
    return quat_rotate(inv, v);
}

Vec3d mat3_mul(const double m[9], Vec3d v) {
    return Vec3d{
        m[0] * v.x + m[1] * v.y + m[2] * v.z,
        m[3] * v.x + m[4] * v.y + m[5] * v.z,
        m[6] * v.x + m[7] * v.y + m[8] * v.z,
    };
}

Vec3d mat3_transpose_mul(const double m[9], Vec3d v) {
    return Vec3d{
        m[0] * v.x + m[3] * v.y + m[6] * v.z,
        m[1] * v.x + m[4] * v.y + m[7] * v.z,
        m[2] * v.x + m[5] * v.y + m[8] * v.z,
    };
}

double clamp01(double value) {
    return std::clamp(value, 0.0, 1.0);
}

uint8_t byte_channel(double value) {
    return static_cast<uint8_t>(std::clamp(value, 0.0, 255.0));
}

void set_pixel(std::vector<uint8_t>* pixels, uint32_t width, uint32_t x, uint32_t y,
        uint8_t r, uint8_t g, uint8_t b) {
    size_t idx = (static_cast<size_t>(y) * width + x) * 3;
    (*pixels)[idx + 0] = r;
    (*pixels)[idx + 1] = g;
    (*pixels)[idx + 2] = b;
}

struct Color {
    double r;
    double g;
    double b;
};

Color mix(Color a, Color b, double t) {
    t = clamp01(t);
    return Color{
        a.r * (1.0 - t) + b.r * t,
        a.g * (1.0 - t) + b.g * t,
        a.b * (1.0 - t) + b.b * t,
    };
}

uint32_t hash_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

double noise01(uint32_t x, uint32_t y, uint64_t sequence_id) {
    uint32_t s = static_cast<uint32_t>(sequence_id * 747796405ull);
    return static_cast<double>(hash_u32(x * 1973u ^ y * 9277u ^ s)) / 4294967295.0;
}

Vec3d camera_ray_to_world(
    const LiftoffRenderDroneState* drone,
    const LiftoffRenderCameraState* camera,
    double u_px,
    double v_px
) {
    double nx = (u_px - camera->cx_px) / std::max(camera->fx_px, 1e-9);
    double ny = (v_px - camera->cy_px) / std::max(camera->fy_px, 1e-9);
    double r2 = nx * nx + ny * ny;
    double inverse_barrel = 1.0 + 0.18 * r2 + 0.035 * r2 * r2;
    Vec3d ray_c = normalize(Vec3d{1.0, nx * inverse_barrel, ny * inverse_barrel});
    Vec3d ray_b = mat3_transpose_mul(camera->body_to_camera, ray_c);
    return normalize(quat_rotate(drone->quat_xyzw, ray_b));
}

Color sky_color(Vec3d ray_w) {
    double up = clamp01(ray_w.z * 0.5 + 0.5);
    Color horizon{98.0, 130.0, 154.0};
    Color zenith{32.0, 54.0, 82.0};
    Color color = mix(horizon, zenith, std::pow(up, 1.7));
    double sun = std::pow(clamp01(dot(normalize(Vec3d{0.55, -0.35, 0.76}), ray_w)), 260.0);
    color = mix(color, Color{255.0, 238.0, 190.0}, sun);
    return color;
}

double grid_line(double value, double spacing, double thickness) {
    double cell = std::abs(std::fmod(value + spacing * 0.5, spacing) - spacing * 0.5);
    return 1.0 - clamp01(cell / thickness);
}

Color ground_color(Vec3d origin_w, Vec3d ray_w) {
    if (std::abs(ray_w.z) <= 1e-9) {
        return Color{76.0, 84.0, 72.0};
    }
    double t = -origin_w.z / ray_w.z;
    if (t <= 0.0) {
        return sky_color(ray_w);
    }
    Vec3d p = origin_w + t * ray_w;
    double checker = std::fmod(std::floor(p.x / 4.0) + std::floor(p.y / 4.0), 2.0);
    Color base = checker == 0.0 ? Color{63.0, 82.0, 66.0} : Color{72.0, 94.0, 72.0};
    double fine = std::max(grid_line(p.x, 1.0, 0.025), grid_line(p.y, 1.0, 0.025));
    double coarse = std::max(grid_line(p.x, 8.0, 0.055), grid_line(p.y, 8.0, 0.055));
    Color color = mix(base, Color{116.0, 126.0, 106.0}, fine * 0.35 + coarse * 0.4);
    double fog = clamp01(t / 120.0);
    return mix(color, Color{100.0, 128.0, 144.0}, fog);
}

Color scene_color(
    const LiftoffRenderDroneState* drone,
    const LiftoffRenderCameraState* camera,
    Vec3d camera_pos_w,
    uint32_t x,
    uint32_t y
) {
    Vec3d ray_w = camera_ray_to_world(
        drone,
        camera,
        static_cast<double>(x) + 0.5,
        static_cast<double>(y) + 0.5
    );
    if (ray_w.z < -0.015) {
        return ground_color(camera_pos_w, ray_w);
    }
    return sky_color(ray_w);
}

void apply_postprocess(
    std::vector<uint8_t>* pixels,
    uint32_t width,
    uint32_t height,
    uint64_t sequence_id
) {
    double cx = (static_cast<double>(width) - 1.0) * 0.5;
    double cy = (static_cast<double>(height) - 1.0) * 0.5;
    double inv_radius = 1.0 / std::max(std::sqrt(cx * cx + cy * cy), 1.0);
    for (uint32_t y = 0; y < height; y++) {
        double scan = 0.965 + 0.035 * std::sin(static_cast<double>(y) * 1.7);
        for (uint32_t x = 0; x < width; x++) {
            size_t idx = (static_cast<size_t>(y) * width + x) * 3;
            double dx = (static_cast<double>(x) - cx) * inv_radius;
            double dy = (static_cast<double>(y) - cy) * inv_radius;
            double rr = dx * dx + dy * dy;
            double vignette = std::clamp(1.08 - 0.42 * rr * rr, 0.45, 1.08);
            double grain = (noise01(x, y, sequence_id) - 0.5) * 7.0;
            double chroma = 1.0 + 0.02 * rr;
            (*pixels)[idx + 0] = byte_channel((*pixels)[idx + 0] * vignette * scan * chroma + grain);
            (*pixels)[idx + 1] = byte_channel((*pixels)[idx + 1] * vignette * scan + grain);
            (*pixels)[idx + 2] = byte_channel((*pixels)[idx + 2] * vignette * scan / chroma + grain);
        }
    }
}

void draw_line(
    std::vector<uint8_t>* pixels,
    uint32_t width,
    uint32_t height,
    int x0,
    int y0,
    int x1,
    int y1,
    Color color
) {
    int dx = std::abs(x1 - x0);
    int sx = x0 < x1 ? 1 : -1;
    int dy = -std::abs(y1 - y0);
    int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    while (true) {
        if (x0 >= 0 && y0 >= 0 && x0 < static_cast<int>(width) && y0 < static_cast<int>(height)) {
            set_pixel(
                pixels,
                width,
                static_cast<uint32_t>(x0),
                static_cast<uint32_t>(y0),
                byte_channel(color.r),
                byte_channel(color.g),
                byte_channel(color.b)
            );
        }
        if (x0 == x1 && y0 == y1) {
            break;
        }
        int e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
    }
}

void draw_disc(
    std::vector<uint8_t>* pixels,
    uint32_t width,
    uint32_t height,
    double cx,
    double cy,
    double radius,
    Color color
) {
    int min_x = std::max(0, static_cast<int>(std::floor(cx - radius)));
    int max_x = std::min(static_cast<int>(width) - 1, static_cast<int>(std::ceil(cx + radius)));
    int min_y = std::max(0, static_cast<int>(std::floor(cy - radius)));
    int max_y = std::min(static_cast<int>(height) - 1, static_cast<int>(std::ceil(cy + radius)));
    double rr_max = radius * radius;
    for (int y = min_y; y <= max_y; y++) {
        for (int x = min_x; x <= max_x; x++) {
            double dx = static_cast<double>(x) + 0.5 - cx;
            double dy = static_cast<double>(y) + 0.5 - cy;
            double rr = dx * dx + dy * dy;
            if (rr > rr_max) {
                continue;
            }
            double edge = std::sqrt(rr) / std::max(radius, 1.0);
            double shade = 1.08 - 0.28 * edge;
            set_pixel(
                pixels,
                width,
                static_cast<uint32_t>(x),
                static_cast<uint32_t>(y),
                byte_channel(color.r * shade),
                byte_channel(color.g * shade),
                byte_channel(color.b * shade)
            );
        }
    }
}

void draw_thick_segment(
    std::vector<uint8_t>* pixels,
    uint32_t width,
    uint32_t height,
    double x0,
    double y0,
    double x1,
    double y1,
    double thickness,
    Color color
) {
    double min_xd = std::min(x0, x1) - thickness;
    double max_xd = std::max(x0, x1) + thickness;
    double min_yd = std::min(y0, y1) - thickness;
    double max_yd = std::max(y0, y1) + thickness;
    int min_x = std::max(0, static_cast<int>(std::floor(min_xd)));
    int max_x = std::min(static_cast<int>(width) - 1, static_cast<int>(std::ceil(max_xd)));
    int min_y = std::max(0, static_cast<int>(std::floor(min_yd)));
    int max_y = std::min(static_cast<int>(height) - 1, static_cast<int>(std::ceil(max_yd)));
    double vx = x1 - x0;
    double vy = y1 - y0;
    double vv = std::max(vx * vx + vy * vy, 1e-9);
    double r2 = thickness * thickness;
    for (int y = min_y; y <= max_y; y++) {
        for (int x = min_x; x <= max_x; x++) {
            double px = static_cast<double>(x) + 0.5;
            double py = static_cast<double>(y) + 0.5;
            double t = clamp01(((px - x0) * vx + (py - y0) * vy) / vv);
            double qx = x0 + t * vx;
            double qy = y0 + t * vy;
            double dx = px - qx;
            double dy = py - qy;
            if (dx * dx + dy * dy > r2) {
                continue;
            }
            set_pixel(
                pixels,
                width,
                static_cast<uint32_t>(x),
                static_cast<uint32_t>(y),
                byte_channel(color.r),
                byte_channel(color.g),
                byte_channel(color.b)
            );
        }
    }
}

void draw_gate_markers(std::vector<uint8_t>* pixels, uint32_t width, uint32_t height) {
    Color color{92.0, 205.0, 214.0};
    int margin_x = static_cast<int>(width / 9);
    int margin_y = static_cast<int>(height / 8);
    int tick = static_cast<int>(std::min(width, height) / 12);
    draw_line(pixels, width, height, margin_x, margin_y, margin_x + tick, margin_y, color);
    draw_line(pixels, width, height, margin_x, margin_y, margin_x, margin_y + tick, color);
    draw_line(pixels, width, height, static_cast<int>(width) - margin_x, margin_y,
              static_cast<int>(width) - margin_x - tick, margin_y, color);
    draw_line(pixels, width, height, static_cast<int>(width) - margin_x, margin_y,
              static_cast<int>(width) - margin_x, margin_y + tick, color);
    draw_line(pixels, width, height, margin_x, static_cast<int>(height) - margin_y,
              margin_x + tick, static_cast<int>(height) - margin_y, color);
    draw_line(pixels, width, height, margin_x, static_cast<int>(height) - margin_y,
              margin_x, static_cast<int>(height) - margin_y - tick, color);
    draw_line(pixels, width, height, static_cast<int>(width) - margin_x, static_cast<int>(height) - margin_y,
              static_cast<int>(width) - margin_x - tick, static_cast<int>(height) - margin_y, color);
    draw_line(pixels, width, height, static_cast<int>(width) - margin_x, static_cast<int>(height) - margin_y,
              static_cast<int>(width) - margin_x, static_cast<int>(height) - margin_y - tick, color);
}

bool project_target(
    const LiftoffRenderCameraState* camera,
    Vec3d pos_c,
    double* u,
    double* v
) {
    if (pos_c.x <= 1e-9 || u == nullptr || v == nullptr) {
        return false;
    }
    double ny = pos_c.y / pos_c.x;
    double nz = pos_c.z / pos_c.x;
    double r2 = ny * ny + nz * nz;
    double barrel = 1.0 - 0.11 * r2 + 0.018 * r2 * r2;
    *u = camera->fx_px * ny * barrel + camera->cx_px;
    *v = camera->fy_px * nz * barrel + camera->cy_px;
    return true;
}

Color material_color(const std::string& name) {
    if (name == "prop") {
        return Color{118.0, 128.0, 132.0};
    }
    if (name == "prop_cyan") {
        return Color{76.0, 150.0, 170.0};
    }
    if (name == "prop_orange") {
        return Color{188.0, 110.0, 48.0};
    }
    if (name == "prop_white") {
        return Color{176.0, 184.0, 178.0};
    }
    if (name == "prop_green") {
        return Color{90.0, 150.0, 92.0};
    }
    if (name == "motor") {
        return Color{34.0, 38.0, 42.0};
    }
    if (name == "battery") {
        return Color{94.0, 116.0, 72.0};
    }
    if (name == "strap") {
        return Color{28.0, 30.0, 34.0};
    }
    if (name == "camera") {
        return Color{42.0, 48.0, 56.0};
    }
    if (name == "camera_blue") {
        return Color{38.0, 62.0, 86.0};
    }
    if (name == "camera_red") {
        return Color{112.0, 48.0, 42.0};
    }
    return Color{58.0, 68.0, 76.0};
}

bool parse_obj_index(const std::string& token, size_t vertex_count, uint32_t* out) {
    if (out == nullptr || token.empty()) {
        return false;
    }
    size_t slash = token.find('/');
    std::string raw = slash == std::string::npos ? token : token.substr(0, slash);
    if (raw.empty()) {
        return false;
    }
    int value = 0;
    try {
        value = std::stoi(raw);
    } catch (...) {
        return false;
    }
    int index = value < 0 ? static_cast<int>(vertex_count) + value : value - 1;
    if (index < 0 || static_cast<size_t>(index) >= vertex_count) {
        return false;
    }
    *out = static_cast<uint32_t>(index);
    return true;
}

std::string target_mesh_path() {
    const char* explicit_path = std::getenv("LIFTOFF_RENDER_DRONE_MESH");
    if (explicit_path != nullptr && explicit_path[0] != '\0') {
        return std::string(explicit_path);
    }
    const char* asset_dir = std::getenv("LIFTOFF_RENDER_ASSET_DIR");
    if (asset_dir != nullptr && asset_dir[0] != '\0') {
        std::string path(asset_dir);
        if (!path.empty() && path.back() != '/') {
            path.push_back('/');
        }
        path += "target_drone.obj";
        return path;
    }
    return ".runs/liftoff_assets/target_drone.obj";
}

void load_target_mesh(LiftoffRenderEngine* engine) {
    if (engine == nullptr || engine->target_mesh_load_attempted) {
        return;
    }
    engine->target_mesh_load_attempted = true;

    std::ifstream in(target_mesh_path());
    if (!in) {
        return;
    }

    std::vector<LiftoffRenderMeshVertex> vertices;
    std::vector<LiftoffRenderMeshTriangle> triangles;
    Color current = material_color("vortex_frame");
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream iss(line);
        std::string tag;
        iss >> tag;
        if (tag == "v") {
            LiftoffRenderMeshVertex vertex{};
            if (iss >> vertex.x >> vertex.y >> vertex.z) {
                vertices.push_back(vertex);
            }
        } else if (tag == "usemtl") {
            std::string material;
            iss >> material;
            current = material_color(material);
        } else if (tag == "f") {
            std::vector<uint32_t> indices;
            std::string token;
            while (iss >> token) {
                uint32_t index = 0;
                if (parse_obj_index(token, vertices.size(), &index)) {
                    indices.push_back(index);
                }
            }
            for (size_t i = 1; i + 1 < indices.size(); i++) {
                triangles.push_back(LiftoffRenderMeshTriangle{
                    indices[0],
                    indices[i],
                    indices[i + 1],
                    byte_channel(current.r),
                    byte_channel(current.g),
                    byte_channel(current.b),
                });
            }
        }
    }

    if (!vertices.empty() && !triangles.empty()) {
        engine->target_mesh_vertices = std::move(vertices);
        engine->target_mesh_triangles = std::move(triangles);
        engine->target_mesh_loaded = true;
    }
}

struct ProjectedVertex {
    double u;
    double v;
    double depth;
    Vec3d world;
    bool valid;
};

ProjectedVertex project_world_vertex(
    const LiftoffRenderDroneState* drone,
    const LiftoffRenderCameraState* camera,
    Vec3d camera_pos_w,
    Vec3d p_w
) {
    Vec3d delta_w = p_w - camera_pos_w;
    Vec3d delta_b = quat_inverse_rotate(drone->quat_xyzw, delta_w);
    Vec3d pos_c = mat3_mul(camera->body_to_camera, delta_b);
    double u = 0.0;
    double v = 0.0;
    bool valid = project_target(camera, pos_c, &u, &v);
    return ProjectedVertex{u, v, pos_c.x, p_w, valid};
}

double edge_value(double ax, double ay, double bx, double by, double px, double py) {
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

void raster_mesh_triangle(
    std::vector<uint8_t>* pixels,
    std::vector<double>* zbuffer,
    uint32_t width,
    uint32_t height,
    const ProjectedVertex& a,
    const ProjectedVertex& b,
    const ProjectedVertex& c,
    Color base,
    bool* wrote_pixel
) {
    if (!a.valid || !b.valid || !c.valid ||
            a.depth <= 1e-6 || b.depth <= 1e-6 || c.depth <= 1e-6) {
        return;
    }

    double min_xd = std::min({a.u, b.u, c.u});
    double max_xd = std::max({a.u, b.u, c.u});
    double min_yd = std::min({a.v, b.v, c.v});
    double max_yd = std::max({a.v, b.v, c.v});
    if (max_xd < 0.0 || max_yd < 0.0 ||
            min_xd >= static_cast<double>(width) || min_yd >= static_cast<double>(height)) {
        return;
    }

    int min_x = std::max(0, static_cast<int>(std::floor(min_xd)));
    int max_x = std::min(static_cast<int>(width) - 1, static_cast<int>(std::ceil(max_xd)));
    int min_y = std::max(0, static_cast<int>(std::floor(min_yd)));
    int max_y = std::min(static_cast<int>(height) - 1, static_cast<int>(std::ceil(max_yd)));
    double area = edge_value(a.u, a.v, b.u, b.v, c.u, c.v);
    if (std::abs(area) <= 1e-9) {
        return;
    }

    Vec3d normal = normalize(cross(b.world - a.world, c.world - a.world));
    Vec3d light = normalize(Vec3d{-0.45, -0.25, 0.86});
    double shade = 0.34 + 0.66 * std::abs(dot(normal, light));
    double fog = clamp01((a.depth + b.depth + c.depth) / 3.0 / 45.0);
    Color shaded = mix(
        Color{base.r * shade, base.g * shade, base.b * shade},
        Color{92.0, 118.0, 134.0},
        fog * 0.25
    );

    for (int y = min_y; y <= max_y; y++) {
        for (int x = min_x; x <= max_x; x++) {
            double px = static_cast<double>(x) + 0.5;
            double py = static_cast<double>(y) + 0.5;
            double w0 = edge_value(b.u, b.v, c.u, c.v, px, py) / area;
            double w1 = edge_value(c.u, c.v, a.u, a.v, px, py) / area;
            double w2 = 1.0 - w0 - w1;
            if (w0 < -1e-5 || w1 < -1e-5 || w2 < -1e-5) {
                continue;
            }
            double depth = w0 * a.depth + w1 * b.depth + w2 * c.depth;
            size_t index = static_cast<size_t>(y) * width + static_cast<uint32_t>(x);
            if (depth <= 1e-6 || depth >= (*zbuffer)[index]) {
                continue;
            }
            (*zbuffer)[index] = depth;
            set_pixel(
                pixels,
                width,
                static_cast<uint32_t>(x),
                static_cast<uint32_t>(y),
                byte_channel(shaded.r),
                byte_channel(shaded.g),
                byte_channel(shaded.b)
            );
            if (wrote_pixel != nullptr) {
                *wrote_pixel = true;
            }
        }
    }
}

bool draw_mesh_target(
    LiftoffRenderEngine* engine,
    std::vector<uint8_t>* pixels,
    std::vector<double>* zbuffer,
    uint32_t width,
    uint32_t height,
    const LiftoffRenderFrameRequest* request,
    const LiftoffRenderCameraState* camera,
    Vec3d camera_pos_w,
    const LiftoffRenderTargetState& target
) {
    load_target_mesh(engine);
    if (engine == nullptr || !engine->target_mesh_loaded) {
        return false;
    }

    Vec3d target_pos_w = as_vec3(target.position_w);
    Vec3d delta_w = target_pos_w - camera_pos_w;
    Vec3d forward_w = as_vec3(target.velocity_w);
    if (norm(forward_w) <= 1e-6) {
        forward_w = -1.0 * delta_w;
    }
    forward_w = normalize(forward_w);
    Vec3d world_up{0.0, 0.0, 1.0};
    Vec3d right_w = cross(world_up, forward_w);
    if (norm(right_w) <= 1e-6) {
        right_w = quat_rotate(request->drone->quat_xyzw, Vec3d{0.0, 1.0, 0.0});
    }
    right_w = normalize(right_w);
    Vec3d up_w = normalize(cross(forward_w, right_w));
    double scale = std::max(target.radius_m, 0.04) / 0.16;

    std::vector<ProjectedVertex> projected;
    projected.reserve(engine->target_mesh_vertices.size());
    for (const LiftoffRenderMeshVertex& vertex : engine->target_mesh_vertices) {
        Vec3d local{vertex.x * scale, vertex.y * scale, vertex.z * scale};
        Vec3d world = target_pos_w + local.x * forward_w + local.y * right_w + local.z * up_w;
        projected.push_back(project_world_vertex(request->drone, camera, camera_pos_w, world));
    }

    bool wrote_pixel = false;
    for (const LiftoffRenderMeshTriangle& tri : engine->target_mesh_triangles) {
        if (tri.a >= projected.size() || tri.b >= projected.size() || tri.c >= projected.size()) {
            continue;
        }
        raster_mesh_triangle(
            pixels,
            zbuffer,
            width,
            height,
            projected[tri.a],
            projected[tri.b],
            projected[tri.c],
            Color{static_cast<double>(tri.r), static_cast<double>(tri.g), static_cast<double>(tri.b8)},
            &wrote_pixel
        );
    }
    return wrote_pixel;
}

void draw_drone_target(
    std::vector<uint8_t>* pixels,
    uint32_t width,
    uint32_t height,
    double u,
    double v,
    double radius_px,
    double heading_rad
) {
    double arm = std::clamp(radius_px * 2.5, 8.0, static_cast<double>(std::min(width, height)) * 0.42);
    double rotor_r = std::clamp(radius_px * 0.48, 2.2, 28.0);
    double body_r = std::clamp(radius_px * 0.55, 2.4, 22.0);
    double thickness = std::clamp(radius_px * 0.17, 1.2, 8.0);
    double ca = std::cos(heading_rad);
    double sa = std::sin(heading_rad);
    double bx = -sa;
    double by = ca;

    double ax0 = u - ca * arm;
    double ay0 = v - sa * arm;
    double ax1 = u + ca * arm;
    double ay1 = v + sa * arm;
    double bx0 = u - bx * arm;
    double by0 = v - by * arm;
    double bx1 = u + bx * arm;
    double by1 = v + by * arm;

    Color shadow{18.0, 24.0, 28.0};
    Color arm_color{42.0, 50.0, 56.0};
    Color rotor_color{24.0, 30.0, 34.0};
    Color body_color{74.0, 86.0, 94.0};
    Color front_color{232.0, 88.0, 52.0};
    Color rear_color{80.0, 178.0, 205.0};

    draw_thick_segment(pixels, width, height, ax0 + 1.5, ay0 + 1.5, ax1 + 1.5, ay1 + 1.5, thickness + 0.8, shadow);
    draw_thick_segment(pixels, width, height, bx0 + 1.5, by0 + 1.5, bx1 + 1.5, by1 + 1.5, thickness + 0.8, shadow);
    draw_thick_segment(pixels, width, height, ax0, ay0, ax1, ay1, thickness, arm_color);
    draw_thick_segment(pixels, width, height, bx0, by0, bx1, by1, thickness, arm_color);

    draw_disc(pixels, width, height, ax0, ay0, rotor_r, rotor_color);
    draw_disc(pixels, width, height, ax1, ay1, rotor_r, rotor_color);
    draw_disc(pixels, width, height, bx0, by0, rotor_r, rotor_color);
    draw_disc(pixels, width, height, bx1, by1, rotor_r, rotor_color);

    draw_disc(pixels, width, height, u, v, body_r, body_color);
    draw_disc(pixels, width, height, u + ca * body_r * 0.7, v + sa * body_r * 0.7,
              std::max(1.5, body_r * 0.32), front_color);
    draw_disc(pixels, width, height, u - ca * body_r * 0.7, v - sa * body_r * 0.7,
              std::max(1.3, body_r * 0.24), rear_color);

    double prop = std::max(2.0, rotor_r * 0.9);
    draw_line(pixels, width, height, static_cast<int>(ax0 - ca * prop), static_cast<int>(ay0 - sa * prop),
              static_cast<int>(ax0 + ca * prop), static_cast<int>(ay0 + sa * prop), Color{108.0, 118.0, 124.0});
    draw_line(pixels, width, height, static_cast<int>(ax1 - ca * prop), static_cast<int>(ay1 - sa * prop),
              static_cast<int>(ax1 + ca * prop), static_cast<int>(ay1 + sa * prop), Color{108.0, 118.0, 124.0});
    draw_line(pixels, width, height, static_cast<int>(bx0 - bx * prop), static_cast<int>(by0 - by * prop),
              static_cast<int>(bx0 + bx * prop), static_cast<int>(by0 + by * prop), Color{108.0, 118.0, 124.0});
    draw_line(pixels, width, height, static_cast<int>(bx1 - bx * prop), static_cast<int>(by1 - by * prop),
              static_cast<int>(bx1 + bx * prop), static_cast<int>(by1 + by * prop), Color{108.0, 118.0, 124.0});
}

LiftoffRenderStatus render_software(
    LiftoffRenderEngine* engine,
    const LiftoffRenderFrameRequest* request,
    LiftoffRenderFrame* frame
) {
    const LiftoffRenderCameraState* camera = request->camera;
    uint32_t width = camera->width_px;
    uint32_t height = camera->height_px;
    if (width == 0 || height == 0 || width > 8192 || height > 8192) {
        return LIFTOFF_RENDER_INVALID_REQUEST;
    }

    engine->pixels.assign(static_cast<size_t>(width) * height * 3, 0);
    std::vector<double> zbuffer(
        static_cast<size_t>(width) * height,
        std::numeric_limits<double>::infinity()
    );
    Vec3d drone_pos_w = as_vec3(request->drone->position_w);
    Vec3d camera_pos_w = drone_pos_w + quat_rotate(
        request->drone->quat_xyzw,
        as_vec3(camera->position_b)
    );

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            Color color = scene_color(request->drone, camera, camera_pos_w, x, y);
            set_pixel(
                &engine->pixels,
                width,
                x,
                y,
                byte_channel(color.r),
                byte_channel(color.g),
                byte_channel(color.b)
            );
        }
    }

    for (uint32_t i = 0; i < request->target_count; i++) {
        const LiftoffRenderTargetState& target = request->targets[i];
        if (draw_mesh_target(
                engine,
                &engine->pixels,
                &zbuffer,
                width,
                height,
                request,
                camera,
                camera_pos_w,
                target)) {
            continue;
        }
        Vec3d delta_w = as_vec3(target.position_w) - camera_pos_w;
        Vec3d delta_b = quat_inverse_rotate(request->drone->quat_xyzw, delta_w);
        Vec3d pos_c = mat3_mul(camera->body_to_camera, delta_b);
        if (pos_c.x <= 1e-9) {
            continue;
        }
        double u = 0.0;
        double v = 0.0;
        if (!project_target(camera, pos_c, &u, &v)) {
            continue;
        }
        if (u < -512.0 || u > static_cast<double>(width) + 512.0 ||
                v < -512.0 || v > static_cast<double>(height) + 512.0) {
            continue;
        }

        double distance = std::max(norm(delta_w), 1e-9);
        double radius_px = std::clamp(
            camera->fx_px * std::max(target.radius_m, 0.03) / distance,
            3.0,
            static_cast<double>(std::max(width, height))
        );
        Vec3d rel_vel_w = as_vec3(target.velocity_w) - as_vec3(request->drone->velocity_w);
        Vec3d rel_vel_b = quat_inverse_rotate(request->drone->quat_xyzw, rel_vel_w);
        Vec3d rel_vel_c = mat3_mul(camera->body_to_camera, rel_vel_b);
        double heading = std::atan2(rel_vel_c.z, rel_vel_c.y);
        if (std::abs(rel_vel_c.y) + std::abs(rel_vel_c.z) < 1e-6) {
            heading = -0.35;
        }
        draw_drone_target(&engine->pixels, width, height, u, v, radius_px, heading);
    }

    apply_postprocess(&engine->pixels, width, height, request->drone->sequence_id);
    draw_gate_markers(&engine->pixels, width, height);

    uint32_t cx = width / 2;
    uint32_t cy = height / 2;
    for (uint32_t dx = 0; dx < std::min<uint32_t>(width / 16, 18); dx++) {
        if (cx + dx < width) set_pixel(&engine->pixels, width, cx + dx, cy, 230, 230, 210);
        if (cx >= dx) set_pixel(&engine->pixels, width, cx - dx, cy, 230, 230, 210);
    }
    for (uint32_t dy = 0; dy < std::min<uint32_t>(height / 16, 18); dy++) {
        if (cy + dy < height) set_pixel(&engine->pixels, width, cx, cy + dy, 230, 230, 210);
        if (cy >= dy) set_pixel(&engine->pixels, width, cx, cy - dy, 230, 230, 210);
    }

    frame->sequence_id = request->drone->sequence_id;
    frame->width_px = width;
    frame->height_px = height;
    frame->channels = 3;
    frame->stride_bytes = width * 3;
    frame->pixels = engine->pixels.data();
    frame->pixel_bytes = engine->pixels.size();
    return LIFTOFF_RENDER_OK;
}

}  // namespace

const char* liftoff_render_status_string(LiftoffRenderStatus status) {
    switch (status) {
        case LIFTOFF_RENDER_OK:
            return "ok";
        case LIFTOFF_RENDER_DISABLED:
            return "rendering disabled";
        case LIFTOFF_RENDER_BACKEND_UNAVAILABLE:
            return "render backend unavailable";
        case LIFTOFF_RENDER_TIMEOUT:
            return "render timed out";
        case LIFTOFF_RENDER_INVALID_REQUEST:
            return "invalid render request";
        case LIFTOFF_RENDER_FRAME_DROPPED:
            return "render frame dropped";
        case LIFTOFF_RENDER_INTERNAL_ERROR:
            return "internal render error";
        default:
            return "unknown render status";
    }
}

LiftoffRenderStatus liftoff_render_engine_create(
    const LiftoffRenderConfig* config,
    LiftoffRenderEngine** engine
) {
    if (engine == nullptr) {
        return LIFTOFF_RENDER_INVALID_REQUEST;
    }
    *engine = nullptr;
    if (config == nullptr) {
        return LIFTOFF_RENDER_INVALID_REQUEST;
    }

    std::unique_ptr<LiftoffRenderEngine> created(new LiftoffRenderEngine{});
    std::memcpy(&created->config, config, sizeof(LiftoffRenderConfig));
    *engine = created.release();
    return LIFTOFF_RENDER_OK;
}

void liftoff_render_engine_destroy(LiftoffRenderEngine* engine) {
    delete engine;
}

LiftoffRenderStatus liftoff_render_frame(
    LiftoffRenderEngine* engine,
    const LiftoffRenderFrameRequest* request,
    LiftoffRenderFrame* frame
) {
    if (frame != nullptr) {
        std::memset(frame, 0, sizeof(LiftoffRenderFrame));
    }
    if (engine == nullptr || request == nullptr || request->drone == nullptr ||
            request->camera == nullptr || frame == nullptr) {
        return LIFTOFF_RENDER_INVALID_REQUEST;
    }
    if (engine->config.backend == LIFTOFF_RENDER_BACKEND_NONE) {
        return LIFTOFF_RENDER_DISABLED;
    }
    if (engine->config.backend == LIFTOFF_RENDER_BACKEND_SOFTWARE) {
        return render_software(engine, request, frame);
    }

    return LIFTOFF_RENDER_BACKEND_UNAVAILABLE;
}

void liftoff_render_release_frame(
    LiftoffRenderEngine* engine,
    LiftoffRenderFrame* frame
) {
    (void)engine;
    if (frame != nullptr) {
        std::memset(frame, 0, sizeof(LiftoffRenderFrame));
    }
}
