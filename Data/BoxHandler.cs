using faceoff.Data.Models;

namespace faceoff.Data {
    public class BoxHandler {
        public class Box {
            public int X;
            public int Y;
            public int height;
            public int width;
            public EmotionType emotion;

            public Box(int x, int y, int h, int w, EmotionType e) {
                this.X = x;
                this.Y = y;
                this.height = h;
                this.width = w;
                this.emotion = e;
            }
        }

        public Box GetBoxData() {
            var rand = new Random();
            var x = rand.Next(200, 280);
            var y = rand.Next(120, 150);
            var h = rand.Next(100, 200);
            var w = rand.Next(100, 200);
            var e = (EmotionType)rand.Next(4);

            Box box = new Box(x, y, h, w, e);
            return box;
        }
    }
}