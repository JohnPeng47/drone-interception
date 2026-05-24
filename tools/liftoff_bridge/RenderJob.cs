using System;
using System.Threading;

namespace DroneInterception.LiftoffBridge
{
    public sealed class RenderJob
    {
        private readonly ManualResetEventSlim done = new ManualResetEventSlim(false);
        private RenderFrameResponse response;
        private Exception error;

        public RenderJob(RenderFrameRequest request)
        {
            Request = request;
        }

        public RenderFrameRequest Request { get; private set; }

        public void Complete(RenderFrameResponse frame, Exception exception)
        {
            response = frame;
            error = exception;
            done.Set();
        }

        public RenderFrameResponse Wait()
        {
            done.Wait();
            if (error != null) throw error;
            if (response == null)
            {
                throw new InvalidOperationException("Render job completed without a frame");
            }
            return response;
        }
    }
}
