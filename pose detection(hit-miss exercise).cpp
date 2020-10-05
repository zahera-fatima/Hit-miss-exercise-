// ----------------------- OpenPose C++ API Tutorial - Example 3 - Body from image -----------------------
// It reads an image, process it, and displays it with the pose (and optionally hand and face) keypoints. In addition,
// it includes all the OpenPose configuration flags (enable/disable hand, face, output saving, etc.).

// Third-party dependencies
#include <opencv2/opencv.hpp>
// Command-line user interface
#define OPENPOSE_FLAGS_DISABLE_PRODUCER
#define OPENPOSE_FLAGS_DISABLE_DISPLAY
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>
#include <cpr/cpr.h>
#include <iostream>
#include <fstream>
#include <json/json.h>
//#include <chrono>
#include <time.h>

using namespace std;

//using namespace std::chrono;
using namespace cv;
Json::Reader reader;
Json::Value obj;
string newaction;
//clock_t start, endpoint;
double fps1 = 0;
int i = 1;

// Custom OpenPose flags
// Producer
DEFINE_string(image_path, "examples/med/pose.jpg",
    "Process an image. Read all standard formats (jpg, png, bmp, etc.).");
// Display
DEFINE_bool(no_display, false,
    "Enable to disable the visual display.");

// This worker will just read and return all the jpg files in a directory
void display(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
{
    try
    {
        // User's displaying/saving/other processing here
            // datum.cvOutputData: rendered frame with pose or heatmaps
            // datum.poseKeypoints: Array<float> with the estimated pose
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            // Display image
            const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumsPtr->at(0)->cvOutputData);
            if (!cvMat.empty())
            {
                cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - Tutorial C++ API", cvMat);
                cv::waitKey(2000);
                cv::destroyAllWindows();
            }
            else
                op::opLog("Empty cv::Mat as output.", op::Priority::High, __LINE__, __FUNCTION__, __FILE__);
        }
        else
            op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

void printKeypoints(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
{

    try
    {
        // Example: How to use the pose keypoints
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            cout << "Detection of Action" << i << " \n" << endl;
            auto x = cpr::Get(cpr::Url{ "https://pose-detection-1-6227a.firebaseio.com/espremote/.json" });
            auto y = x.text;
            reader.parse(y, obj);
            auto espdata = obj["value"].asString();
            if (espdata == "1")
            {
                cout << "Left Hand Up of Robosapien." << endl;
            }
            if (espdata == "2")
            {
                cout << "Right Hand Up of Robosapien." << endl;
            }
            if (espdata == "3")
            {
                cout << "Both Hand Up of Robosapien." << endl;
            }
            if (espdata == "4")
            {
                cout << "Both Hand Down of Robosapien." << endl;
            }
            //cout << espdata << endl;
            const auto& poseKeypoints = datumsPtr->at(0)->poseKeypoints;
            for (auto person = 0; person < poseKeypoints.getSize(0); person++)
            {
                auto y2 = poseKeypoints[{0, 2, 1}];
                auto y3 = poseKeypoints[{0, 3, 1}];
                auto y4 = poseKeypoints[{0, 4, 1}];
                auto y5 = poseKeypoints[{0, 5, 1}];
                auto y6 = poseKeypoints[{0, 6, 1}];
                auto y7 = poseKeypoints[{0, 7, 1}];

                //op::opLog("y2 = " + std::to_string(y2) + " y3 = " + std::to_string(y3) + " y4 = " + std::to_string(y4), op::Priority::High);
                //op::opLog("y5 = " + std::to_string(y5) + " y6 = " + std::to_string(y6) + " y7 = " + std::to_string(y7), op::Priority::High);
                if (y2 - y3 >= 0 && y3 - y4 > 0 || y5 - y6 >= 0 && y6 - y7 > 0)
                {
                    //opLog("right hands up" , op::Priority::High);
                    if (y2 - y3 >= 0 && y3 - y4 > 0 && y5 - y6 >= 0 && y6 - y7 > 0)
                    {
                        opLog("both hands up", op::Priority::High);
                        string action = "2";
                        if (espdata == action)
                        {
                            cout << "Action perform by Child successfully! It marked as 'HIT' ." << endl;
                        }
                        else
                        {
                            cout << "Child Miss the Action." << endl;
                        }
                        break;
                    }
                    if (y5 - y6 >= 0 && y6 - y7 > 0)
                    {
                        opLog("left hands up", op::Priority::High);
                        string action = "1";

                        if (espdata == action)
                        {
                            cout << "Action perform by Child successfully! It marked as 'HIT' ." << endl;
                        }
                        else
                        {
                            cout << "Child Miss the Action." << endl;
                        }
                    }
                    if (y2 - y3 >= 0 && y3 - y4 > 0)
                    {
                        opLog("right hands up", op::Priority::High);


                        string action = "3";


                        if (espdata == action)
                        {
                            cout << "Action perform by Child successfully! It marked as 'HIT' ." << endl;
                        }
                        else
                        {
                            cout << "Child Miss the Action." << endl;
                        }
                    }

                }
                else {
                    opLog("both hands down", op::Priority::High);

                    string action = "4";
                    if (espdata == action)
                    {
                        cout << "Action perform by Child successfully! It marked as 'HIT' ." << endl;
                    }
                    else
                    {
                        cout << "Child Miss the Action." << endl;
                    }
                }
            }
        }
        else
            op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
    i = i++;
    cout << "\n";
    cout << "\n";
}

void configureWrapper(op::Wrapper& opWrapper)
{
    try
    {
        // Configuring OpenPose

        // logging_level
        op::checkBool(
            0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
            __LINE__, __FUNCTION__, __FILE__);
        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
        op::Profiler::setDefaultX(FLAGS_profile_speed);

        // Applying user defined configuration - GFlags to program variables
        // outputSize
        const auto outputSize = op::flagsToPoint(op::String(FLAGS_output_resolution), "-1x-1");
        // netInputSize
        const auto netInputSize = op::flagsToPoint(op::String(FLAGS_net_resolution), "-1x368");

        // poseMode
        const auto poseMode = op::flagsToPoseMode(FLAGS_body);
        // poseModel
        const auto poseModel = op::flagsToPoseModel(op::String(FLAGS_model_pose));
        // JSON saving
        if (!FLAGS_write_keypoint.empty())
            op::opLog(
                "Flag `write_keypoint` is deprecated and will eventually be removed. Please, use `write_json`"
                " instead.", op::Priority::Max);
        // keypointScaleMode
        const auto keypointScaleMode = op::flagsToScaleMode(FLAGS_keypoint_scale);
        // heatmaps to add
        const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
            FLAGS_heatmaps_add_PAFs);
        const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
        // >1 camera view?
        const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1);
        // Face and hand detectors
        const auto faceDetector = op::flagsToDetector(FLAGS_face_detector);
        const auto handDetector = op::flagsToDetector(FLAGS_hand_detector);
        // Enabling Google Logging
        const bool enableGoogleLogging = true;

        // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
        const op::WrapperStructPose wrapperStructPose{
            poseMode, netInputSize, outputSize, keypointScaleMode, FLAGS_num_gpu, FLAGS_num_gpu_start,
            FLAGS_scale_number, (float)FLAGS_scale_gap, op::flagsToRenderMode(FLAGS_render_pose, multipleView),
            poseModel, !FLAGS_disable_blending, (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap,
            FLAGS_part_to_show, op::String(FLAGS_model_folder), heatMapTypes, heatMapScaleMode, FLAGS_part_candidates,
            (float)FLAGS_render_threshold, FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max,
            op::String(FLAGS_prototxt_path), op::String(FLAGS_caffemodel_path),
            (float)FLAGS_upsampling_ratio, enableGoogleLogging };
        opWrapper.configure(wrapperStructPose);
        // Extra functionality configuration (use op::WrapperStructExtra{} to disable it)
        const op::WrapperStructExtra wrapperStructExtra{
            FLAGS_3d, FLAGS_3d_min_views, FLAGS_identification, FLAGS_tracking, FLAGS_ik_threads };
        opWrapper.configure(wrapperStructExtra);
        // Output (comment or use default argument to disable any output)
        const op::WrapperStructOutput wrapperStructOutput{
            FLAGS_cli_verbose, op::String(FLAGS_write_keypoint), op::stringToDataFormat(FLAGS_write_keypoint_format),
            op::String(FLAGS_write_json), op::String(FLAGS_write_coco_json), FLAGS_write_coco_json_variants,
            FLAGS_write_coco_json_variant, op::String(FLAGS_write_images), op::String(FLAGS_write_images_format),
            op::String(FLAGS_write_video), FLAGS_write_video_fps, FLAGS_write_video_with_audio,
            op::String(FLAGS_write_heatmaps), op::String(FLAGS_write_heatmaps_format), op::String(FLAGS_write_video_3d),
            op::String(FLAGS_write_video_adam), op::String(FLAGS_write_bvh), op::String(FLAGS_udp_host),
            op::String(FLAGS_udp_port) };
        opWrapper.configure(wrapperStructOutput);
        // No GUI. Equivalent to: opWrapper.configure(op::WrapperStructGui{});
        // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
        if (FLAGS_disable_multi_thread)
            opWrapper.disableMultiThreading();
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

int tutorialApiCpp()
{
    try
    {

        op::opLog("PESONALIZED ROBOT MEDIATED IMITATION THERAPY FOR AUTISTIC CHILDREN", op::Priority::High);
        const auto opTimer = op::getTimerInit();

        // Configuring OpenPose
        op::opLog("Analysis of Child Actions", op::Priority::High);
        op::Wrapper opWrapper{ op::ThreadManagerMode::Asynchronous };
        configureWrapper(opWrapper);

        // Starting OpenPose
        op::opLog("Hit Miss Report", op::Priority::High);
        opWrapper.start();


        VideoCapture cap(0);

        // Check if camera opened successfully

        if (!cap.isOpened()) {

            cout << "Error opening video stream or file" << endl;
            return -1;

        }
        //auto start = high_resolution_clock::now();
        //auto duration = duration_cast<milliseconds>(start);
        //start = clock();

        while (1) {
            //start = clock();
            double fps = cap.get(CAP_PROP_FPS);
            //cout << fps << endl;


            Mat frame;

            // Capture frame-by-frame

            cap.read(frame);


            /*
            // If the frame is empty, break immediately

            if (frame.empty())
                break;*/
                // Display the resulting frame

            imshow("camera", frame);

            // Press  ESC on keyboard to exit

            //waitKey(100);
            char c = (char)waitKey(25);


            if (c == 's' || fps1 == 7680) {
                //waitKey(2000);
                Mat frame2;
                frame2 = frame.clone();
                //imshow("captured frame", frame2);
                imwrite("D:/ROBOSAPIEN/openpose-master/examples/med/pose.jpg", frame2);
                // Process and display image

                const cv::Mat cvImageToProcess = cv::imread(FLAGS_image_path);
                const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(cvImageToProcess);
                auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);
                if (datumProcessed != nullptr)
                {
                    printKeypoints(datumProcessed);
                    if (!FLAGS_no_display)
                        display(datumProcessed);
                }
                else {
                    op::opLog("Image could not be processed.", op::Priority::High);
                }
                fps1 = 0;
            }



            if (c == 27) {
                break;
            }
            //endpoint = clock();
            //auto stop = high_resolution_clock::now();
            fps1 = fps1 + fps;
            //cout << fps1 << endl;

            //auto duration = duration_cast<milliseconds>(stop - start);
            //cout << "Time taken by function: "
                //<< duration.count() << " milliseconds" << endl;
            //auto start = stop;
            /*double duration = double(endpoint-start) / CLOCKS_PER_SEC;
            cout << duration << endl;
            addpoint = double(duration + addpoint) / CLOCKS_PER_SEC;
            cout << addpoint << endl;
            start = endpoint;*/

        }



        // Measuring total time
        op::printTime(opTimer, "OpenPose demo successfully finished. Total time: ", " seconds.", op::Priority::High);


        cap.release();
        // Closes all the frames
        destroyAllWindows();
        // Return
        return 0;
    }

    catch (const std::exception&)
    {
        return -1;
    }
}

int main(int argc, char* argv[])
{
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running tutorialApiCpp
    return tutorialApiCpp();
}
