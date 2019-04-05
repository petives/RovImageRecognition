using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Util;


namespace RovShapeRecognition
{
    public partial class FormShapeDetection : Form
    {
        Image<Bgr, byte> imgInput;
        public FormShapeDetection()
        {
            InitializeComponent();
        }

        private void FormShapeDetection_Load(object sender, EventArgs e)
        {
            panel1.AutoScroll = true;
        }

        private void openToolStripMenuItem_Click(object sender, EventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog();
            try
            {
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    imgInput = new Image<Bgr, byte>(dialog.FileName);
                    pictureBox1.SizeMode = PictureBoxSizeMode.AutoSize;
                    pictureBox1.Image = imgInput.Bitmap;
                }
            }
            catch (Exception)
            {
                MessageBox.Show("File format is not image.");
            }
        }
        bool playing = false;
        VideoCapture capture = new VideoCapture();
        Mat mCap = new Mat();
        private void startToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (!playing) //just to initialize only once
            {
                playing = true;
                pictureBox1.Width = capture.Width;
                pictureBox1.Height = capture.Height;
            }
            capture.ImageGrabbed += Capture_ImageGrabbed;
            capture.Start();
            capture.SetCaptureProperty(Emgu.CV.CvEnum.CapProp.FrameWidth, 640);
            capture.SetCaptureProperty(Emgu.CV.CvEnum.CapProp.FrameHeight, 540);
        }

        private void stopToolStripMenuItem_Click(object sender, EventArgs e)
        {
            playing = false;
            capture.Stop();
        }

        private void Capture_ImageGrabbed(object sender, EventArgs e)
        {
            capture.Read(mCap);
            try
            {
                Image<Bgr, byte> capImage = new Image<Bgr, byte>(mCap.Bitmap);
                CvInvoke.Flip(capImage, capImage, Emgu.CV.CvEnum.FlipType.Horizontal);
                findShapes(capImage);
            }
            catch(Exception)
            {
                MessageBox.Show("Something went wrong...\nTry starting the webcam when an image is not already loaded.");
                Application.Exit();
            }
        }

        private void detectShapesToolStripMenuItem_Click_1(object sender, EventArgs e)
        {
            if (imgInput == null)
            {
                MessageBox.Show("No image detected.");
                return;
            }
            findShapes(imgInput);
        }

        public int Get(Mat mat, int index)
        {
            if (mat.Depth != Emgu.CV.CvEnum.DepthType.Cv32S)
            {
                throw new ArgumentOutOfRangeException("ContourData must have Cv32S hierarchy element type.");
            }
            if (mat.Rows != 1)
            {
                throw new ArgumentOutOfRangeException("ContourData must have one hierarchy hierarchy row.");
            }
            if (mat.NumberOfChannels != 4)
            {
                throw new ArgumentOutOfRangeException("ContourData must have four hierarchy channels.");
            }
            if (mat.Dims != 2)
            {
                throw new ArgumentOutOfRangeException("ContourData must have two dimensional hierarchy.");
            }
            long elementStride = mat.ElementSize / sizeof(Int32);
            var offset = (long)mat + index * elementStride;
            if (0 <= offset && offset < Hierarchy.Total.ToInt64() * elementStride)
            {
                unsafe
                {
                    return *((Int32*)Hierarchy.DataPointer.ToPointer() + offset);
                }
            }
            else
            {
                return -1;
            }
        }

        private void findShapes(Image<Bgr, byte> imgInput)
        {
            Image<Gray, byte> imgCanny = new Image<Gray, byte>(imgInput.Width, imgInput.Height, new Gray(0));
            imgCanny = imgInput.Canny(20, 50);

            //pictureBox1.Image = imgCanny.Bitmap;

            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            Mat mat = new Mat();
            CvInvoke.FindContours(imgCanny, contours, mat, Emgu.CV.CvEnum.RetrType.Tree, Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxSimple);
            
            int[] shapes = new int[4];
            /* [0] . . . Triangle
             * [1] . . . Square
             * [2] . . . Rectangle
             * [3] . . . Circle
             */

            for (int i = 0; i < contours.Size; i++)
            {
                double perimeter = CvInvoke.ArcLength(contours[i], true);
                VectorOfPoint approx = new VectorOfPoint();
                CvInvoke.ApproxPolyDP(contours[i], approx, 0.04 * perimeter, true);

                CvInvoke.DrawContours(imgInput, contours, i, new MCvScalar(0, 0, 255), 2);
                pictureBox1.Image = imgInput.Bitmap;
                //moments  center of the shape

                var moments = CvInvoke.Moments(contours[i]);
                int x = (int)(moments.M10 / moments.M00);
                int y = (int)(moments.M01 / moments.M00);

                if (CvInvoke.ContourArea(approx) > 250) //ONLY CONSIDERS SHAPES WITH AREA GREATER THAN 100
                {
                    if (approx.Size == 3)
                    {
                        shapes[0]++;
                        CvInvoke.PutText(imgInput, "Triangle", new Point(x, y),
                            Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.5, new MCvScalar(0, 255, 0), 1);
                    }

                    if (approx.Size == 4)
                    {
                        Rectangle rect = CvInvoke.BoundingRectangle(contours[i]);

                        double ar = (double)rect.Width / rect.Height;

                        if (ar >= 0.95 && ar <= 1.05)
                        {
                            shapes[1]++;
                            CvInvoke.PutText(imgInput, "Square", new Point(x, y),
                            Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.5, new MCvScalar(0, 255, 0), 1);
                        }
                        else
                        {
                            shapes[2]++;
                            CvInvoke.PutText(imgInput, "Rectangle", new Point(x, y),
                            Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.5, new MCvScalar(0, 255, 0), 1);
                        }

                    }

                    if (approx.Size > 6)
                    {
                        CvInvoke.PutText(imgInput, "Circle", new Point(x, y),
                        Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.5, new MCvScalar(0, 255, 0), 1);
                        shapes[3]++;
                    }

                    pictureBox1.Image = imgInput.Bitmap;
                }
            }
            textBox8.Text = shapes[0].ToString();
            textBox6.Text = shapes[1].ToString();
            textBox4.Text = shapes[2].ToString();
            textBox2.Text = shapes[3].ToString();
        }
    }
}


