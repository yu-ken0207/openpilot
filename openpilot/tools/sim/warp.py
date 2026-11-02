import platform
from typing import Any
import numpy as np
import pyray as rl

from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

GL_PACK_ROW_LENGTH, GL_RGBA, GL_UNSIGNED_BYTE = 0x0D02, 0x1908, 0x1401
# the camera thread may still be sending the previous frame while the next one renders
N_BUFFERS = 3

# NV12 chroma sample (i, j) sits at luma pixel (2i + 0.5, 2j + 0.5)
LUMA_FROM_CHROMA = np.array([[2., 0., 0.5], [0., 2., 0.5], [0., 0., 1.]])


def warp_params(K: np.ndarray, view_from_calib: np.ndarray, height: float, ds: float, e: float, psi: float):
  """Inverse warp from the simulated to the logged camera: p_log ~ G @ p + g * max(m @ p, 0).

  The simulated camera sits ds ahead, e to the right and yawed psi to the right of the logged one.
  Rays that hit the road (m @ p > 0) see the full relative pose, rays above the horizon only see the rotation."""
  A = view_from_calib @ rot_from_euler([0., 0., psi]) @ view_from_calib.T
  b = view_from_calib @ np.array([ds, e, 0.]) / height
  n = view_from_calib[:, 2]
  K_inv = np.linalg.inv(K)
  return K @ A @ K_inv, K @ b, K_inv.T @ n


def chroma_params(G: np.ndarray, g: np.ndarray, m: np.ndarray):
  S, S_inv = LUMA_FROM_CHROMA, np.linalg.inv(LUMA_FROM_CHROMA)
  return S_inv @ G @ S, S_inv @ g, S.T @ m


GLSL_VERSION = "#version 330 core\n" if platform.system() == "Darwin" else "#version 300 es\nprecision highp float;\n"

VERTEX_SHADER = GLSL_VERSION + """
in vec3 vertexPosition;
uniform mat4 mvp;
void main() {
  gl_Position = mvp * vec4(vertexPosition, 1.0);
}
"""

# Each output texel packs 4 bytes of a plane row: 4 luma samples, or 2 interleaved chroma pairs
FRAGMENT_SHADER = GLSL_VERSION + """
uniform sampler2D texture0;
uniform vec2 srcSize;
uniform vec3 G[3];
uniform vec3 g;
uniform vec3 m;
uniform int chroma;
out vec4 fragColor;

vec4 sample_at(float x, float y) {
  vec3 p = vec3(x, y, 1.0);
  vec3 s = vec3(dot(G[0], p), dot(G[1], p), dot(G[2], p)) + g * max(dot(m, p), 0.0);
  vec2 src = clamp(s.xy / s.z, vec2(0.0), srcSize - 1.0);
  return texture(texture0, (src + 0.5) / srcSize);
}

void main() {
  vec2 o = floor(gl_FragCoord.xy);
  if (chroma == 0) {
    fragColor = vec4(sample_at(4.0*o.x, o.y).r, sample_at(4.0*o.x + 1.0, o.y).r, sample_at(4.0*o.x + 2.0, o.y).r, sample_at(4.0*o.x + 3.0, o.y).r);
  } else {
    fragColor = vec4(sample_at(2.0*o.x, o.y).ra, sample_at(2.0*o.x + 1.0, o.y).ra);
  }
}
"""


class Warp:
  """Inverse warps a packed NV12 frame into a VisionIPC-strided NV12 buffer with a fragment shader, clamping to the frame edge.
  raylib falls back to llvmpipe without a GPU. All Warps must be created and called from the thread that called init()."""
  shader: rl.Shader
  locs: dict[str, int]
  read_pixels: Any
  pixel_store: Any

  @staticmethod
  def init() -> None:
    rl.set_trace_log_level(rl.TraceLogLevel.LOG_WARNING)
    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)
    rl.init_window(64, 64, "sim warp")
    assert rl.is_window_ready(), "no OpenGL context to warp frames in"
    Warp.shader = rl.load_shader_from_memory(VERTEX_SHADER, FRAGMENT_SHADER)
    assert rl.is_shader_valid(Warp.shader), "warp shader failed to compile"
    Warp.locs = {k: rl.get_shader_location(Warp.shader, k) for k in ("srcSize", "G", "g", "m", "chroma")}
    # rlReadTexturePixels allocates a new buffer for every read, so frames are read straight into the VisionIPC layout instead
    Warp.read_pixels = rl.ffi.cast("void (*)(int, int, int, int, unsigned int, unsigned int, void *)", rl.rl_get_proc_address("glReadPixels"))
    Warp.pixel_store = rl.ffi.cast("void (*)(unsigned int, int)", rl.rl_get_proc_address("glPixelStorei"))
    assert Warp.read_pixels != rl.ffi.NULL and Warp.pixel_store != rl.ffi.NULL, "no glReadPixels"

  def __init__(self, w: int, h: int):
    self.w, self.h = w, h
    self.stride, y_height, _, size = get_nv12_info(w, h)
    self.uv_offset = self.stride * y_height
    self.bufs = [np.zeros(size, dtype=np.uint8) for _ in range(N_BUFFERS)]
    self.buf_idx = 0
    self.planes = []
    for pw, ph, fmt in ((w, h, rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAYSCALE), (w // 2, h // 2, rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAY_ALPHA)):
      tex = rl.Texture(rl.rl_load_texture(rl.ffi.NULL, pw, ph, fmt, 1), pw, ph, 1, fmt)
      rl.set_texture_filter(tex, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
      rl.set_texture_wrap(tex, rl.TextureWrap.TEXTURE_WRAP_CLAMP)
      self.planes.append((tex, rl.load_render_texture(w // 4, ph)))

  def __call__(self, nv12: np.ndarray, params) -> np.ndarray:
    w, h, stride = self.w, self.h, self.stride
    self.buf_idx = (self.buf_idx + 1) % N_BUFFERS
    buf = self.bufs[self.buf_idx]
    if params is None:
      buf[:stride * h].reshape(h, stride)[:, :w] = nv12[:w * h].reshape(h, w)
      buf[self.uv_offset:self.uv_offset + stride * h // 2].reshape(h // 2, stride)[:, :w] = nv12[w * h:].reshape(h // 2, w)
      return buf

    rl.rl_set_blend_factors(1, 0, 0x8006)  # GL_ONE, GL_ZERO, GL_FUNC_ADD: write samples as is
    for chroma, ((tex, target), data, offset, (G, g, m)) in enumerate(zip(self.planes, (nv12[:w * h], nv12[w * h:]), (0, self.uv_offset),
                                                                          (params, chroma_params(*params)), strict=True)):
      rl.rl_update_texture(tex.id, 0, 0, tex.width, tex.height, tex.format, rl.ffi.cast("void *", rl.ffi.from_buffer(data)))
      rl.begin_texture_mode(target)
      rl.begin_shader_mode(self.shader)
      rl.begin_blend_mode(rl.BlendMode.BLEND_CUSTOM)
      rl.set_shader_value(self.shader, self.locs["srcSize"], rl.Vector2(tex.width, tex.height), rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2)
      for k, v in (("G", G), ("g", g), ("m", m)):
        vec3s = rl.ffi.new("float[]", v.ravel().tolist())
        rl.set_shader_value_v(self.shader, self.locs[k], vec3s, rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3, v.size // 3)
      rl.set_shader_value(self.shader, self.locs["chroma"], rl.ffi.new("int *", chroma), rl.ShaderUniformDataType.SHADER_UNIFORM_INT)
      rl.draw_texture_pro(tex, rl.Rectangle(0, 0, tex.width, tex.height), rl.Rectangle(0, 0, target.texture.width, target.texture.height),
                          rl.Vector2(0, 0), 0., rl.WHITE)
      rl.end_blend_mode()
      rl.end_shader_mode()
      rl.end_texture_mode()
      # a texel is 4 bytes of a plane row, so rows land stride // 4 texels apart
      rl.rl_enable_framebuffer(target.id)
      self.pixel_store(GL_PACK_ROW_LENGTH, stride // 4)
      self.read_pixels(0, 0, target.texture.width, target.texture.height, GL_RGBA, GL_UNSIGNED_BYTE, rl.ffi.from_buffer(buf[offset:]))
      self.pixel_store(GL_PACK_ROW_LENGTH, 0)
      rl.rl_disable_framebuffer()
    return buf
