namespace faceoff.Data {
    public class ImageHandler {
        public void SaveBinary(byte[] data, String path) {
            Console.WriteLine($"Data received (before Base64 decode): {data}");
            using var writer = new BinaryWriter(File.OpenWrite(path));
            writer.Write(data);
        }
    }
}