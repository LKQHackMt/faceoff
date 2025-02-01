using FaceONNX;
namespace faceoff.Core
{

    public class FaceOffService
    {


        public FaceOffService()
        {


        }

        public void CameraImageFeed(byte[] imageDataArray)
        {
            //aggregate the face and emotion data
            //if face is detected, call FaceDetectionModel
            //then call EmotionRecognition
            //else 
            var currImageData = imageDataArray;
            if(currImageData != null)
            {
                Console.WriteLine("Image data received");
                FaceDetectionModel(currImageData);
                EmotionRecognition(currImageData);
            }
            else
            {
                Console.WriteLine("Image data not received");

                return;
            }
        }

        public void FaceDetectionModel(byte[] imageDataArray)
        {
            // dummy face coordinates
            var faceCoordinates = new List<(int X, int Y, int Width, int Height)>
            {
                (50, 50, 100, 100),
                (200, 200, 150, 150)
            };

            foreach (var face in faceCoordinates)
            {
                Console.WriteLine($"Face detected at X: {face.X}, Y: {face.Y}, Width: {face.Width}, Height: {face.Height}");
            }
            


            //return face coordinates 
        }

        private void EmotionRecognition(byte[] imageDataArray)
        {
            //confidence level threshold

            //return emotion;
        }

    }
}

    