namespace faceoff.Data {
    public class BoxHandler {
        public class Box {
            public int X = 50;
            public int Y = 50;
            public int height = 50;
            public int width = 50;
        }
        public Box GetBoxData() {
            Box box = new Box();
            return box;
        }
    }
}