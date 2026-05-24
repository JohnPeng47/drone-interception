using System;
using System.Reflection;
using UnityEngine;

namespace DroneInterception.LiftoffBridge
{
    public sealed class LiftoffCameraBinder : MonoBehaviour
    {
        private Component twinCamera;
        private Camera primaryCamera;
        private RenderTexture renderTexture;
        private Texture2D readback;

        public RenderFrameResponse Render(RenderFrameRequest request)
        {
            BindCamera();
            ApplyPose(request);
            ApplyFov(request);

            int width = request.Camera.WidthPx;
            int height = request.Camera.HeightPx;
            EnsureTargets(width, height);

            RenderTexture previous = RenderTexture.active;
            RenderTexture previousTarget = primaryCamera.targetTexture;
            try
            {
                primaryCamera.targetTexture = renderTexture;
                primaryCamera.Render();
                RenderTexture.active = renderTexture;
                readback.ReadPixels(new Rect(0, 0, width, height), 0, 0, false);
                readback.Apply(false);
                return new RenderFrameResponse(width, height, 3, readback.GetRawTextureData());
            }
            finally
            {
                primaryCamera.targetTexture = previousTarget;
                RenderTexture.active = previous;
            }
        }

        private void BindCamera()
        {
            if (primaryCamera != null) return;

            Type twinType = Type.GetType("TwinCamera, Assembly-CSharp");
            if (twinType != null)
            {
                UnityEngine.Object[] twins = FindObjectsOfType(twinType);
                if (twins.Length > 0)
                {
                    twinCamera = twins[0] as Component;
                    PropertyInfo prop = twinType.GetProperty("PrimaryCam", BindingFlags.Public | BindingFlags.Instance);
                    if (prop != null)
                    {
                        primaryCamera = prop.GetValue(twinCamera, null) as Camera;
                    }
                }
            }

            if (primaryCamera == null)
            {
                throw new InvalidOperationException("Could not bind Liftoff TwinCamera.PrimaryCam");
            }
        }

        private void ApplyPose(RenderFrameRequest request)
        {
            Quaternion vehicleRotation = SimQuatToUnity(request.Vehicle.QuatXyzw);
            Vector3 cameraOffset = SimVecToUnity(request.Camera.PositionB);
            Vector3 position = SimVecToUnity(request.Vehicle.PositionW) + vehicleRotation * cameraOffset;
            Quaternion rotation = CameraRotationToUnity(vehicleRotation, request.Camera.BodyToCamera);

            primaryCamera.transform.SetPositionAndRotation(position, rotation);
        }

        private static Vector3 SimVecToUnity(float[] v)
        {
            return new Vector3(v[0], v[2], v[1]);
        }

        private static Quaternion SimQuatToUnity(float[] q)
        {
            return new Quaternion(q[0], q[2], q[1], q[3]);
        }

        private static Quaternion CameraRotationToUnity(Quaternion vehicleRotation, float[,] bodyToCamera)
        {
            Vector3 cameraXBody = new Vector3(bodyToCamera[0, 0], bodyToCamera[0, 2], bodyToCamera[0, 1]);
            Vector3 cameraYBody = new Vector3(bodyToCamera[1, 0], bodyToCamera[1, 2], bodyToCamera[1, 1]);
            Vector3 cameraZBody = new Vector3(bodyToCamera[2, 0], bodyToCamera[2, 2], bodyToCamera[2, 1]);

            Vector3 unityForward = vehicleRotation * cameraXBody.normalized;
            Vector3 unityRight = vehicleRotation * cameraYBody.normalized;
            Vector3 unityUp = vehicleRotation * cameraZBody.normalized;
            if (unityForward.sqrMagnitude < 1e-8f || unityUp.sqrMagnitude < 1e-8f)
            {
                return vehicleRotation;
            }
            return Quaternion.LookRotation(unityForward, unityUp);
        }

        private void ApplyFov(RenderFrameRequest request)
        {
            float verticalFovDeg = request.Camera.VfovRad * Mathf.Rad2Deg;
            if (twinCamera != null)
            {
                MethodInfo setFov = twinCamera.GetType().GetMethod("SetFOV", BindingFlags.Public | BindingFlags.Instance);
                if (setFov != null)
                {
                    setFov.Invoke(twinCamera, new object[] { verticalFovDeg });
                    return;
                }
            }
            primaryCamera.fieldOfView = verticalFovDeg;
        }

        private void EnsureTargets(int width, int height)
        {
            if (renderTexture != null && renderTexture.width == width && renderTexture.height == height) return;
            if (renderTexture != null) renderTexture.Release();
            renderTexture = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
            readback = new Texture2D(width, height, TextureFormat.RGB24, false);
        }
    }
}
