using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace faceoff.Core
{
    public sealed class DetectedFace
    {
        public float Confidence { get; set; }
        public float X { get; set; }
        public float Y { get; set; }
        public float Width { get; set; }
        public float Height { get; set; }
    }

    internal struct Anchor
    {
        public float CenterX;
        public float CenterY;
        public float Width;
        public float Height;
    }

    public sealed class FaceOffService : IDisposable
    {
        // Define the network input size (should match your model)
        private const int InputWidth = 320;
        private const int InputHeight = 240;

        // These names should match your model’s input and output node names.
        private const string InputName = "input";
        private const string OutputNameBbox = "boxes";
        private const string OutputNameConf = "scores";

        // These variance values are used to decode bounding boxes (should match training)
        private static readonly float[] Variances = { 0.1f, 0.1f, 0.2f, 0.2f };

        private readonly InferenceSession _session;
        private readonly List<Anchor> _anchors;

        public FaceOffService(string onnxModelPath)
        {
            _session = new InferenceSession(onnxModelPath);
            _anchors = GenerateAnchors();
        }

        public List<DetectedFace> CameraImageFeed(byte[] imageDataArray)
        {
            //aggregate the face and emotion data
            //if face is detected, call FaceDetectionModel
            //then call EmotionRecognition
            //else 
            var currImageData = imageDataArray;


            if (currImageData != null)
            {
                Console.WriteLine("Image data received");
                var DetectedFace = DetectFaces(currImageData);
                //EmotionRecognition(currImageData);
                return DetectedFace;
            }
            else
            {
                Console.WriteLine("Image data not received");
                return null;
            }
            
        }

        //public List<DetectedFace> FaceDetectionModel(byte[] imageDataArray)
        //{

        //    var DetectedFace = DetectFaces(imageDataArray);
        //    // dummy face coordinates

        //    return DetectedFace;
        //}
        /// <summary>
        /// Run the detection:
        ///   1. Preprocess the image
        ///   2. Evaluate the model
        ///   3. Decode outputs and apply NMS
        /// </summary>
        /// 
        public List<DetectedFace> DetectFaces(byte[] imageData, float confidenceThreshold = 0.7f)
        {
            using Image<Rgb24> originalImage = Image.Load<Rgb24>(imageData);
            int originalWidth = originalImage.Width;
            int originalHeight = originalImage.Height;

            DenseTensor<float> inputTensor = PreprocessImage(imageData);

            var inputs = new List<NamedOnnxValue>
    {
        NamedOnnxValue.CreateFromTensor(InputName, inputTensor)
    };

            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);
            var bboxOut = results.FirstOrDefault(o => o.Name == OutputNameBbox)?.AsTensor<float>();
            var confOut = results.FirstOrDefault(o => o.Name == OutputNameConf)?.AsTensor<float>();

            if (bboxOut == null || confOut == null)
                return new List<DetectedFace>();

            float[] bboxArray = bboxOut.ToArray();
            float[] confArray = confOut.ToArray();

            var faces = DecodeAndNms(bboxArray, confArray, _anchors, confidenceThreshold);

            // Scale bounding boxes back to the original image size
            float scaleX = (float)originalWidth / InputWidth;
            float scaleY = (float)originalHeight / InputHeight;

            foreach (var face in faces)
            {
                face.X *= scaleX;
                face.Y *= scaleY;
                face.Width *= scaleX;
                face.Height *= scaleY;
            }

            return faces;
        }

        /// <summary>
        /// Preprocess the image:
        ///   - Load and resize to 320x240
        ///   - Build a tensor with shape [1,3,240,320] in RGB order
        /// </summary>
        private DenseTensor<float> PreprocessImage(byte[] imageData)
        {
            using Image<Rgb24> image = Image.Load<Rgb24>(imageData);
            image.Mutate(ctx => ctx.Resize(InputWidth, InputHeight));

            var tensor = new DenseTensor<float>(new[] { 1, 3, InputHeight, InputWidth });
            for (int y = 0; y < InputHeight; y++)
            {
                for (int x = 0; x < InputWidth; x++)
                {
                    Rgb24 pixel = image[x, y];
                    // Channel 0: R, Channel 1: G, Channel 2: B
                    tensor[0, 0, y, x] = pixel.R / 255.0f;
                    tensor[0, 1, y, x] = pixel.G / 255.0f;
                    tensor[0, 2, y, x] = pixel.B / 255.0f;

                }
            }
            return tensor;
        }

        /// <summary>
        /// Generate anchors based on the image size, strides and min-box values.
        /// Adjust these values to match your training setup.
        /// </summary>
        private List<Anchor> GenerateAnchors()
        {
            var anchors = new List<Anchor>();

            // Use the same strides and box sizes as in the original implementation
            int[] strides = { 8, 16, 32, 64 };
            // Each feature map has a set of “min boxes”
            float[][] minBoxes = new float[][]
            {
                new float[] { 10f, 16f, 24f },
                new float[] { 32f, 48f },
                new float[] { 64f, 96f },
                new float[] { 128f, 192f, 256f }
            };

            for (int i = 0; i < strides.Length; i++)
            {
                int stride = strides[i];
                // Compute feature map dimensions (using Ceiling to match the Emgu.CV version)
                int fmWidth = (int)Math.Ceiling((float)InputWidth / stride);
                int fmHeight = (int)Math.Ceiling((float)InputHeight / stride);
                for (int y = 0; y < fmHeight; y++)
                {
                    for (int x = 0; x < fmWidth; x++)
                    {
                        // Normalize the center coordinates to [0,1]
                        float centerX = (x + 0.5f) / fmWidth;
                        float centerY = (y + 0.5f) / fmHeight;
                        foreach (var box in minBoxes[i])
                        {
                            // Normalize box dimensions by image size
                            float w = box / (float)InputWidth;
                            float h = box / (float)InputHeight;
                            anchors.Add(new Anchor
                            {
                                CenterX = centerX,
                                CenterY = centerY,
                                Width = w,
                                Height = h
                            });
                        }
                    }
                }
            }
            return anchors;
        }

        /// <summary>
        /// Decode the output bounding boxes using the generated anchors and variances.
        /// Then apply non-max suppression.
        /// </summary>
        private List<DetectedFace> DecodeAndNms(
            float[] loc, float[] conf, List<Anchor> anchors,
            float confThreshold, float nmsIouThreshold = 0.1f)
        {
            var faces = new List<DetectedFace>();
            int anchorCount = anchors.Count;
            const int boxParams = 4; // [dx, dy, dw, dh]

            for (int i = 0; i < anchorCount; i++)
            {
                // Perform softmax on the two class scores
                float background = conf[i * 2];
                float faceScore = conf[i * 2 + 1];
                float sum = (float)(Math.Exp(background) + Math.Exp(faceScore));
                float score = (float)Math.Exp(faceScore) / sum;

                if (score < confThreshold)
                    continue;

                float dx = loc[i * boxParams];
                float dy = loc[i * boxParams + 1];
                float dw = loc[i * boxParams + 2];
                float dh = loc[i * boxParams + 3];

                Anchor anchor = anchors[i];

                // Decode bounding box center and dimensions
                float predCenterX = anchor.CenterX + dx * Variances[0] * anchor.Width;
                float predCenterY = anchor.CenterY + dy * Variances[1] * anchor.Height;
                float predWidth = (float)Math.Exp(dw * Variances[2]) * anchor.Width;
                float predHeight = (float)Math.Exp(dh * Variances[3]) * anchor.Height;

                // Convert to corner coordinates
                float xMin = predCenterX - predWidth / 2f;
                float yMin = predCenterY - predHeight / 2f;
                float xMax = predCenterX + predWidth / 2f;
                float yMax = predCenterY + predHeight / 2f;

                // Ensure coordinates remain within the valid image range
                xMin = Math.Max(0, xMin * InputWidth);
                yMin = Math.Max(0, yMin * InputHeight);
                xMax = Math.Min(InputWidth, xMax * InputWidth);
                yMax = Math.Min(InputHeight, yMax * InputHeight);


                float width = xMax - xMin;
                float height = yMax - yMin;
                if (width < 5 || height < 5)
                    continue;

                faces.Add(new DetectedFace
                {
                    Confidence = score,
                    X = xMin,
                    Y = yMin,
                    Width = width,
                    Height = height
                });
            }

            return NonMaxSuppression(faces, nmsIouThreshold);
        }

        /// <summary>
        /// Simple non-max suppression implementation.
        /// </summary>
        private List<DetectedFace> NonMaxSuppression(List<DetectedFace> faces, float iouThreshold)
        {
            var sorted = faces.OrderByDescending(f => f.Confidence).ToList();
            var result = new List<DetectedFace>();

            foreach (var current in sorted)
            {
                bool keep = true;
                foreach (var kept in result)
                {
                    if (IoU(current, kept) > iouThreshold)
                    {
                        keep = false;
                        break;
                    }
                }
                if (keep)
                    result.Add(current);
            }
            return result;
        }

        /// <summary>
        /// Compute the Intersection over Union (IoU) of two boxes.
        /// </summary>
        private float IoU(DetectedFace a, DetectedFace b)
        {
            float x1 = Math.Max(a.X, b.X);
            float y1 = Math.Max(a.Y, b.Y);
            float x2 = Math.Min(a.X + a.Width, b.X + b.Width);
            float y2 = Math.Min(a.Y + a.Height, b.Y + b.Height);
            float intersection = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
            float union = a.Width * a.Height + b.Width * b.Height - intersection;
            return union <= 0 ? 0 : intersection / union;
        }

        public void Dispose() => _session.Dispose();
    }
}
