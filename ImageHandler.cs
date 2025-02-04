using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.IO;

namespace faceoff.Core
{
    public class ImageHandler2
    {
        /// <summary>
        /// Crops the face region from the full image.
        /// </summary>
        /// <param name="imageData">The full image as a byte array.</param>
        /// <param name="face">The detected face containing the crop coordinates.</param>
        /// <returns>The cropped face image as a PNG byte array.</returns>
        public byte[] CropFace(byte[] imageData, DetectedFace face)
        {
            using (var image = Image.Load<Rgb24>(imageData))
            {
                // Create a rectangle based on the detected face's coordinates.
                var cropRect = new Rectangle(
                    (int)face.X,
                    (int)face.Y,
                    (int)face.Width,
                    (int)face.Height
                );

                // Ensure the rectangle is within the bounds of the image.
                cropRect.Intersect(new Rectangle(0, 0, image.Width, image.Height));

                image.Mutate(ctx => ctx.Crop(cropRect));

                using (var ms = new MemoryStream())
                {
                    // Save the cropped image as PNG.
                    image.SaveAsPng(ms);
                    return ms.ToArray();
                }
            }
        }
    }
}
