using System;
using System.Globalization;
using Newtonsoft.Json.Linq;

namespace DroneInterception.LiftoffBridge
{
    public static class RenderProtocol
    {
        public static RenderFrameRequest ParseRequest(string json)
        {
            JObject root = JObject.Parse(json);
            JObject vehicle = (JObject)root["vehicle_state"];
            JObject camera = (JObject)root["camera"];
            return new RenderFrameRequest(
                FloatArray(vehicle, "position_w", 3),
                FloatArray(vehicle, "quat_xyzw", 4),
                new RenderCamera(
                    IntValue(camera, "width_px"),
                    IntValue(camera, "height_px"),
                    FloatValue(camera, "vfov_rad"),
                    FloatArray(camera, "position_b", 3),
                    FloatMatrix(camera, "body_to_camera", 3, 3)
                )
            );
        }

        public static string FrameInfoJson(RenderFrameResponse response)
        {
            return "{\"width_px\":" + response.WidthPx.ToString(CultureInfo.InvariantCulture)
                + ",\"height_px\":" + response.HeightPx.ToString(CultureInfo.InvariantCulture)
                + ",\"channels\":" + response.Channels.ToString(CultureInfo.InvariantCulture)
                + "}";
        }

        private static float[] FloatArray(JObject obj, string key, int length)
        {
            JArray valuesJson = (JArray)obj[key];
            if (valuesJson.Count != length) throw new InvalidOperationException("Invalid " + key);
            float[] values = new float[length];
            for (int i = 0; i < length; i++)
            {
                values[i] = valuesJson[i].Value<float>();
            }
            return values;
        }

        private static float[,] FloatMatrix(JObject obj, string key, int rows, int cols)
        {
            JArray valuesJson = (JArray)obj[key];
            if (valuesJson.Count != rows) throw new InvalidOperationException("Invalid " + key);
            float[,] values = new float[rows, cols];
            for (int r = 0; r < rows; r++)
            {
                JArray row = (JArray)valuesJson[r];
                if (row.Count != cols) throw new InvalidOperationException("Invalid " + key);
                for (int c = 0; c < cols; c++)
                {
                    values[r, c] = row[c].Value<float>();
                }
            }
            return values;
        }

        private static int IntValue(JObject obj, string key)
        {
            return obj[key].Value<int>();
        }

        private static float FloatValue(JObject obj, string key)
        {
            return obj[key].Value<float>();
        }
    }

    public sealed class RenderFrameRequest
    {
        public RenderFrameRequest(float[] positionW, float[] quatXyzw, RenderCamera camera)
        {
            Vehicle = new RenderVehicle(positionW, quatXyzw);
            Camera = camera;
        }

        public RenderVehicle Vehicle { get; private set; }
        public RenderCamera Camera { get; private set; }
    }

    public sealed class RenderVehicle
    {
        public RenderVehicle(float[] positionW, float[] quatXyzw)
        {
            PositionW = positionW;
            QuatXyzw = quatXyzw;
        }

        public float[] PositionW { get; private set; }
        public float[] QuatXyzw { get; private set; }
    }

    public sealed class RenderCamera
    {
        public RenderCamera(int widthPx, int heightPx, float vfovRad, float[] positionB, float[,] bodyToCamera)
        {
            WidthPx = widthPx;
            HeightPx = heightPx;
            VfovRad = vfovRad;
            PositionB = positionB;
            BodyToCamera = bodyToCamera;
        }

        public int WidthPx { get; private set; }
        public int HeightPx { get; private set; }
        public float VfovRad { get; private set; }
        public float[] PositionB { get; private set; }
        public float[,] BodyToCamera { get; private set; }
    }

    public sealed class RenderFrameResponse
    {
        public RenderFrameResponse(int widthPx, int heightPx, int channels, byte[] pixels)
        {
            WidthPx = widthPx;
            HeightPx = heightPx;
            Channels = channels;
            Pixels = pixels;
        }

        public int WidthPx { get; private set; }
        public int HeightPx { get; private set; }
        public int Channels { get; private set; }
        public byte[] Pixels { get; private set; }
    }
}
