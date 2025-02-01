namespace faceoff.Data {
    // This class exists to provide functions to send the images to the backend
    public class ImageHandler {
        public void SaveBinary(byte[] data, String path) {
            using var writer = new BinaryWriter(File.OpenWrite(path));
            writer.Write(data);
        }
    }
}