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
    private const int EmotionInputSize = 48;
    private readonly InferenceSession _session;
    // Updated emotion labels to match the PyTorch model
    private readonly string[] _emotionLabels = new[]
    {
        "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
    };

    public EmotionDetectionService(string modelPath)
    {
       
        _session = new InferenceSession(modelPath);

        // Debug: Print model information
        Console.WriteLine("\nModel Information:");
        foreach (var input in _session.InputMetadata)
        {
            Console.WriteLine($"Input Name: {input.Key}");
            Console.WriteLine($"Input Shape: [{string.Join(",", input.Value.Dimensions)}]");
            Console.WriteLine($"Input Type: {input.Value.ElementType}");
        }

        Console.WriteLine("\nOutput Information:");
        foreach (var output in _session.OutputMetadata)
        {
            Console.WriteLine($"Output Name: {output.Key}");
            Console.WriteLine($"Output Shape: [{string.Join(",", output.Value.Dimensions)}]");
            Console.WriteLine($"Output Type: {output.Value.ElementType}");
        }
    }

    public async Task<EmotionResult> DetectEmotion(DetectedFace face, byte[] originalImageData)
    {
        using var image = Image.Load<Rgb24>(originalImageData);

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
        if (right > image.Width)
        {
            left -= (right - image.Width);
            right = image.Width;
        }
        if (bottom > image.Height)
        {
            top -= (bottom - image.Height);
            bottom = image.Height;
        }

        // Ensure left and top are not negative after adjustments
        left = Math.Max(0, left);
        top = Math.Max(0, top);

        // Calculate final width and height
        var width = Math.Min(right - left, image.Width - left);
        var height = Math.Min(bottom - top, image.Height - top);

        // Create rectangle with validated coordinates
        var rectangle = new Rectangle(
            (int)left,
            (int)top,
            (int)width,
            (int)height
        );

        // Debug output
        Console.WriteLine($"Image dimensions: {image.Width}x{image.Height}");
        Console.WriteLine($"Face detection: X={face.X}, Y={face.Y}, Width={face.Width}, Height={face.Height}");
        Console.WriteLine($"Crop rectangle: X={rectangle.X}, Y={rectangle.Y}, Width={rectangle.Width}, Height={rectangle.Height}");

        // Verify rectangle is valid before cropping
        if (rectangle.Width <= 0 || rectangle.Height <= 0 ||
            rectangle.Right > image.Width || rectangle.Bottom > image.Height)
        {
            throw new ArgumentException($"Invalid crop rectangle: {rectangle}");
        }

        // Process image with validated rectangle
        using var faceImage = image.Clone(ctx => ctx
            .Crop(rectangle)
            .Grayscale()
            .Resize(EmotionInputSize, EmotionInputSize));

        // Create tensor [1, 1, 48, 48]
        var tensor = new DenseTensor<float>(new[] { 1, 1, EmotionInputSize, EmotionInputSize });

        // Normalize values from [0, 255] to [-1, 1]
        const float SCALE = 1.0f / 255.0f;

        for (int y = 0; y < EmotionInputSize; y++)
        {
            for (int x = 0; x < EmotionInputSize; x++)
            {
                var pixel = faceImage[x, y];
                float normalized = (pixel.R * SCALE * 2.0f) - 1.0f;
                tensor[0, 0, y, x] = normalized;
            }
        }

        // Get the correct input name and run inference
        string inputName = _session.InputMetadata.Keys.First();
        var inputs = new List<NamedOnnxValue>
    {
        NamedOnnxValue.CreateFromTensor(inputName, tensor)
    };

        using var results = _session.Run(inputs);
        var output = results.First().AsTensor<float>();

        // Process results
        int numEmotions = output.Dimensions[1];
        var probabilities = new float[numEmotions];
        for (int i = 0; i < numEmotions; i++)
        {
            probabilities[i] = output[0, i];
        }

        probabilities = Softmax(probabilities);

        // Log probabilities
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

    // ExtractModel method remains the same...
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