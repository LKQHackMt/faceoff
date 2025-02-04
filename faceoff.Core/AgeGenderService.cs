using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.IO;

namespace faceoff.Core
{
    #region Prediction Result Classes

    // Result for the age prediction model.
    public class AgePredictionResult
    {
        public float Age { get; set; }
        public float Confidence { get; set; }

    }

    // Result for the gender prediction model.
    public class GenderPredictionResult
    {
        public string Gender { get; set; }
        public float Confidence { get; set; }
    }

    // Combined result for a detected face.
    public class EnhancedFaceDetection
    {
        public DetectedFace Face { get; set; }
        public AgePredictionResult AgePrediction { get; set; }
        public GenderPredictionResult GenderPrediction { get; set; }
    }

    #endregion

    #region Age Prediction Service



    #endregion

    #region Gender Prediction Service

    public sealed class AgePredictionService : IDisposable
    {
        private const int InputWidth = 224;
        private const int InputHeight = 224;
        private const string InputName = "input";
        private const string AgeOutputName = "age_prob";

        // Updated age ranges based on the model's training data
        private static readonly float[] AgeRanges = new float[]
        {
        1,     // 0-2
        5,     // 3-7
        12,    // 8-16
        23,    // 17-29
        35,    // 30-40
        45,    // 41-49
        58,    // 50-66
        75     // 67+
        };

        private readonly InferenceSession _session;

        public AgePredictionService(string onnxModelPath)
        {
            _session = new InferenceSession(onnxModelPath);
        }

        public async Task<AgePredictionResult> PredictAgeAsync(byte[] faceImageData)
        {
            try
            {
                DenseTensor<float> inputTensor = await Task.Run(() => PreprocessImage(faceImageData));

                var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(InputName, inputTensor)
        };

                using var results = await Task.Run(() => _session.Run(inputs));
                var probabilities = results.First().AsTensor<float>();

                if (probabilities == null)
                    return null;

                float[] probs = probabilities.ToArray();

                int maxIndex1 = 0, maxIndex2 = 1;
                if (probs[1] > probs[0]) { maxIndex1 = 1; maxIndex2 = 0; }

                for (int i = 2; i < probs.Length; i++)
                {
                    if (probs[i] > probs[maxIndex1])
                    {
                        maxIndex2 = maxIndex1;
                        maxIndex1 = i;
                    }
                    else if (probs[i] > probs[maxIndex2])
                    {
                        maxIndex2 = i;
                    }
                }

                float age1 = AgeRanges[maxIndex1];
                float age2 = AgeRanges[maxIndex2];
                float prob1 = probs[maxIndex1];
                float prob2 = probs[maxIndex2];
                float totalProb = prob1 + prob2;

                float predictedAge = (age1 * prob1 + age2 * prob2) / totalProb;

                if (predictedAge > 20) predictedAge *= 1.15f;

                return new AgePredictionResult
                {
                    Age = predictedAge,
                    Confidence = probs[maxIndex1]
                };
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in PredictAge: {ex.Message}");
                return null;
            }
        }

        private DenseTensor<float> PreprocessImage(byte[] imageData)
        {
            using Image<Rgb24> image = Image.Load<Rgb24>(imageData);
            image.Mutate(ctx => ctx.Resize(InputWidth, InputHeight));

            var tensor = new DenseTensor<float>(new[] { 1, 3, InputHeight, InputWidth });

            // Caffe-style preprocessing
            float[] meanValues = new float[] { 104.0f, 117.0f, 123.0f };

            for (int y = 0; y < InputHeight; y++)
            {
                for (int x = 0; x < InputWidth; x++)
                {
                    Rgb24 pixel = image[x, y];
                    // BGR order for CaffeNet
                    tensor[0, 0, y, x] = pixel.B - meanValues[0];
                    tensor[0, 1, y, x] = pixel.G - meanValues[1];
                    tensor[0, 2, y, x] = pixel.R - meanValues[2];
                }
            }
            return tensor;
        }

        public void Dispose() => _session?.Dispose();
    }

    public sealed class GenderPredictionService : IDisposable
    {
        private const int InputWidth = 224;
        private const int InputHeight = 224;
        private const string InputName = "input";
        private const string GenderOutputName = "gender_prob";
        private static readonly string[] GenderClasses = { "Male", "Female" }; // Swapped order to match model

        private readonly InferenceSession _session;

        public GenderPredictionService(string onnxModelPath)
        {
            _session = new InferenceSession(onnxModelPath);
        }

        public async Task<GenderPredictionResult> PredictGenderAsync(byte[] faceImageData)
        {
            try
            {
                DenseTensor<float> inputTensor = await Task.Run(() => PreprocessImage(faceImageData));

                var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(InputName, inputTensor)
        };

                using var results = await Task.Run(() => _session.Run(inputs));
                var genderOutput = results.First().AsTensor<float>();

                if (genderOutput == null)
                    return null;

                float[] genderScores = genderOutput.ToArray();
                float[] genderProbs = Softmax(genderScores);
                int genderIndex = genderProbs[0] > genderProbs[1] ? 0 : 1;

                return new GenderPredictionResult
                {
                    Gender = GenderClasses[genderIndex],
                    Confidence = genderProbs[genderIndex]
                };
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in PredictGender: {ex.Message}");
                return null;
            }
        }

        private DenseTensor<float> PreprocessImage(byte[] imageData)
        {
            using Image<Rgb24> image = Image.Load<Rgb24>(imageData);
            image.Mutate(ctx => ctx.Resize(InputWidth, InputHeight));

            var tensor = new DenseTensor<float>(new[] { 1, 3, InputHeight, InputWidth });

            // Caffe-style preprocessing
            float[] meanValues = new float[] { 104.0f, 117.0f, 123.0f };

            for (int y = 0; y < InputHeight; y++)
            {
                for (int x = 0; x < InputWidth; x++)
                {
                    Rgb24 pixel = image[x, y];
                    // BGR order for CaffeNet
                    tensor[0, 0, y, x] = pixel.B - meanValues[0];
                    tensor[0, 1, y, x] = pixel.G - meanValues[1];
                    tensor[0, 2, y, x] = pixel.R - meanValues[2];
                }
            }
            return tensor;
        }

        private float[] Softmax(float[] scores)
        {
            float max = scores.Max();
            float[] exp = scores.Select(x => (float)Math.Exp(x - max)).ToArray();
            float sum = exp.Sum();
            return exp.Select(x => x / sum).ToArray();
        }

        public void Dispose() => _session?.Dispose();
    }
    #endregion

    #region FaceOffService Extensions

    public static class FaceOffServiceExtensions
    {
        /// <summary>
        /// For each detected face, crop the face image and run the age and gender prediction models.
        /// </summary>
        public static async Task<List<EnhancedFaceDetection>> DetectFacesWithAgeGender(
            this FaceOffService faceService,
            AgePredictionService ageService,
            GenderPredictionService genderService,
            byte[] imageData,
            float confidenceThreshold = 0.7f)
        {
            // Detect faces using your existing face detector.
            var detectedFaces = faceService.DetectFaces(imageData, confidenceThreshold);
            var results = new List<EnhancedFaceDetection>();

            // Load the full image once.
            using (Image<Rgb24> originalImage = Image.Load<Rgb24>(imageData))
            {
                foreach (var face in detectedFaces)
                {
                    // Define the face crop rectangle.
                    Rectangle faceRect = new Rectangle(
                        (int)face.X,
                        (int)face.Y,
                        (int)face.Width,
                        (int)face.Height
                    );

                    // Crop the face image.
                    using (var faceImage = originalImage.Clone(ctx => ctx.Crop(faceRect)))
                    {
                        using (var ms = new MemoryStream())
                        {
                            await faceImage.SaveAsPngAsync(ms);
                            byte[] faceBytes = ms.ToArray();

                            // Run the age and gender prediction models separately.
                            var ageResult =await ageService.PredictAgeAsync(faceBytes);
                            var genderResult =await genderService.PredictGenderAsync(faceBytes);

                            results.Add(new EnhancedFaceDetection
                            {
                                Face = face,
                                AgePrediction = ageResult,
                                GenderPrediction = genderResult
                            });
                        }
                    }
                }
            }

            return results;
        }
    }

    #endregion
}
