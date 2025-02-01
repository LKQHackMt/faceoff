using Azure;
using Azure.AI.Vision.Face;

namespace faceoff.Core
{

    public class FaceOffCore
    {

        public FaceOffCore()
        {


        }

        public void ProcessImageData(byte[] imageDataArray)
        {
            //aggregate the face and emotion data
            //if face is detected, call FaceDetectionModel
            //then call EmotionRecognition
            //else 
        }

        private void FaceDetectionModel(byte[] imageDataArray)
        {
            //return face coordinates
        }

        private void EmotionRecognition(byte[] imageDataArray)
        {
            //confidence level threshold
            //return emotion;
        }

    }
}

    