using faceoff.Core;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.IO.Compression;
using System.Linq;

public class EmotionResult
{
    public string Emotion { get; set; }
    public float Confidence { get; set; }
}

public class EmotionDetectionService : IDisposable
{
    // Update constants for the new model
    private const int EmotionInputSize = 224;  // Changed from 48 to 224
    private const int InputChannels = 3;       // Changed from 1 to 3

    private readonly InferenceSession _session;
    private readonly string[] _emotionLabels = new[]
    {
        "angry", "contempt", "disgusted", "fear", "happy", "neutral", "sad", "surprise"
    };

    // EfficientNet normalization values (ImageNet statistics)
    private readonly float[] ChannelMeans = new float[] { 0.485f, 0.456f, 0.406f };
    private readonly float[] ChannelStds = new float[] { 0.229f, 0.224f, 0.225f };

    public EmotionDetectionService(string modelPath)
    {
        _session = new InferenceSession(modelPath);

        // Print model information for debugging
        Console.WriteLine("\nModel Information:");
        foreach (var input in _session.InputMetadata)
        {
            Console.WriteLine($"Input Name: {input.Key}");
            Console.WriteLine($"Input Shape: [{string.Join(",", input.Value.Dimensions)}]");
            Console.WriteLine($"Input Type: {input.Value.ElementType}");
        }
    }

    public async Task<EmotionResult> DetectEmotion(DetectedFace face, byte[] originalImageData)
    {
        using var image = Image.Load<Rgb24>(originalImageData);

        // Your existing face cropping logic remains the same
        var rectangle = CalculateFaceCropRectangle(face, image.Width, image.Height);

        // Process image: crop -> resize (no grayscale conversion)
        using var faceImage = image.Clone(ctx => ctx
            .Crop(rectangle)
            .Resize(EmotionInputSize, EmotionInputSize));

        // Create tensor with new dimensions [1, 3, 224, 224]
        var tensor = new DenseTensor<float>(new[] { 1, InputChannels, EmotionInputSize, EmotionInputSize });

        // Fill tensor with normalized RGB values
        for (int y = 0; y < EmotionInputSize; y++)
        {
            for (int x = 0; x < EmotionInputSize; x++)
            {
                var pixel = faceImage[x, y];

                // Normalize each channel separately using ImageNet statistics
                tensor[0, 0, y, x] = (pixel.R / 255f - ChannelMeans[0]) / ChannelStds[0];  // R channel
                tensor[0, 1, y, x] = (pixel.G / 255f - ChannelMeans[1]) / ChannelStds[1];  // G channel
                tensor[0, 2, y, x] = (pixel.B / 255f - ChannelMeans[2]) / ChannelStds[2];  // B channel
            }
        }

        // Create input with the correct name from your model
        string inputName = _session.InputMetadata.Keys.First();
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, tensor)
        };

        // Run inference
        using var results = _session.Run(inputs);
        var output = results.First().AsTensor<float>();

        // Process results
        int numEmotions = output.Dimensions[1];
        var probabilities = new float[numEmotions];
        for (int i = 0; i < numEmotions; i++)
        {
            probabilities[i] = output[0, i];
        }

        // Apply softmax to get probabilities
        probabilities = Softmax(probabilities);

        // Log probabilities for debugging
        Console.WriteLine("\nEmotion probabilities:");
        for (int i = 0; i < Math.Min(numEmotions, _emotionLabels.Length); i++)
        {
            Console.WriteLine($"{_emotionLabels[i]}: {probabilities[i]:P2}");
        }

        int maxIndex = Array.IndexOf(probabilities, probabilities.Max());

        return new EmotionResult
        {
            Emotion = maxIndex < _emotionLabels.Length ? _emotionLabels[maxIndex] : $"emotion_{maxIndex}",
            Confidence = probabilities[maxIndex]
        };
    }

    private Rectangle CalculateFaceCropRectangle(DetectedFace face, int imageWidth, int imageHeight)
    {
        // Calculate face center
        var centerX = face.X + (face.Width / 2);
        var centerY = face.Y + (face.Height / 2);

        // Calculate desired crop size (1.4x the face size)
        var desiredSize = Math.Max(face.Width, face.Height) * 1.4f;

        // Calculate initial crop boundaries
        var left = centerX - (desiredSize / 2);
        var top = centerY - (desiredSize / 2);
        var right = left + desiredSize;
        var bottom = top + desiredSize;

        // Adjust boundaries to fit within image
        if (left < 0)
        {
            right += Math.Abs(left);
            left = 0;
        }
        if (top < 0)
        {
            bottom += Math.Abs(top);
            top = 0;
        }
        if (right > imageWidth)
        {
            left -= (right - imageWidth);
            right = imageWidth;
        }
        if (bottom > imageHeight)
        {
            top -= (bottom - imageHeight);
            bottom = imageHeight;
        }

        // Ensure coordinates are within bounds
        left = Math.Max(0, left);
        top = Math.Max(0, top);

        var width = Math.Min(right - left, imageWidth - left);
        var height = Math.Min(bottom - top, imageHeight - top);

        return new Rectangle((int)left, (int)top, (int)width, (int)height);
    }

    private float[] Softmax(float[] logits)
    {
        float maxLogit = logits.Max();
        var expValues = logits.Select(x => Math.Exp(x - maxLogit)).ToArray();
        float sumExp = (float)expValues.Sum();
        return expValues.Select(x => (float)(x / sumExp)).ToArray();
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}
// Simple TAR reader implementation
public class TarReader : IDisposable
{
    private readonly Stream _stream;

    public TarReader(Stream stream)
    {
        _stream = stream;
    }

    public void ReadToEnd(string destinationPath)
    {
        byte[] buffer = new byte[100];

        while (true)
        {
            // Read header
            var header = _stream.Read(buffer, 0, 100);
            if (header == 0 || IsEmptyBlock(buffer)) break;

            // Parse filename (first 100 bytes)
            string filename = GetString(buffer).Trim('\0');
            if (string.IsNullOrEmpty(filename)) break;

            // Skip to file size (124 bytes from start)
            _stream.Seek(24, SeekOrigin.Current);

            // Read file size (12 bytes)
            byte[] sizeBuffer = new byte[12];
            _stream.Read(sizeBuffer, 0, 12);
            var size = Convert.ToInt64(GetString(sizeBuffer).Trim().TrimEnd('\0'), 8);

            // Skip rest of header
            _stream.Seek(376, SeekOrigin.Current);

            // Extract file
            string destinationFile = Path.Combine(destinationPath, filename);
            Directory.CreateDirectory(Path.GetDirectoryName(destinationFile));

            using (var fileStream = File.Create(destinationFile))
            {
                CopyStream(_stream, fileStream, size);
            }

            // Move to next 512-byte boundary
            var remainder = size % 512;
            if (remainder > 0)
            {
                _stream.Seek(512 - remainder, SeekOrigin.Current);
            }
        }
    }

    private static bool IsEmptyBlock(byte[] buffer)
    {
        return buffer.All(b => b == 0);
    }

    private static string GetString(byte[] buffer)
    {
        return System.Text.Encoding.ASCII.GetString(buffer);
    }

    private static void CopyStream(Stream input, Stream output, long bytes)
    {
        byte[] buffer = new byte[32768];
        int read;
        while (bytes > 0 && (read = input.Read(buffer, 0, (int)Math.Min(buffer.Length, bytes))) > 0)
        {
            output.Write(buffer, 0, read);
            bytes -= read;
        }
    }

    public void Dispose()
    {
        _stream?.Dispose();
    }
}