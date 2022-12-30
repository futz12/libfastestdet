#include "pch.h"

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

struct _fastestdet
{
    ncnn::Net net;

    std::vector<unsigned char> param;
    std::vector<unsigned char> model;
};

typedef _fastestdet* __fastestdet;

float Sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

float Tanh(float x)
{
    return 2.0f / (1.0f + exp(-2 * x)) - 1;
}

class TargetBox
{
private:
    float GetWidth() { return (x2 - x1); };
    float GetHeight() { return (y2 - y1); };

public:
    int x1;
    int y1;
    int x2;
    int y2;

    int category;
    float score;

    float area() { return GetWidth() * GetHeight(); };
};

float IntersectionArea(const TargetBox& a, const TargetBox& b)
{
    if (a.x1 > b.x2 || a.x2 < b.x1 || a.y1 > b.y2 || a.y2 < b.y1)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
    float inter_height = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);

    return inter_width * inter_height;
}

bool scoreSort(TargetBox a, TargetBox b)
{
    return (a.score > b.score);
}

//NMS处理
int nmsHandle(std::vector<TargetBox>& src_boxes, std::vector<TargetBox>& dst_boxes, const float nms_threshold)
{
    std::vector<int> picked;

    sort(src_boxes.begin(), src_boxes.end(), scoreSort);

    for (int i = 0; i < src_boxes.size(); i++)
    {
        int keep = 1;
        for (int j = 0; j < picked.size(); j++)
        {
            //交集
            float inter_area = IntersectionArea(src_boxes[i], src_boxes[picked[j]]);
            //并集
            float union_area = src_boxes[i].area() + src_boxes[picked[j]].area() - inter_area;
            float IoU = inter_area / union_area;

            if (IoU > nms_threshold && src_boxes[i].category == src_boxes[picked[j]].category)
            {
                keep = 0;
                break;
            }
        }

        if (keep) {
            picked.push_back(i);
        }
    }

    for (int i = 0; i < picked.size(); i++)
    {
        dst_boxes.push_back(src_boxes[picked[i]]);
    }

    return 0;
}


extern "C" __declspec(dllexport) __fastestdet __stdcall fastestdet_Init(const unsigned char* mem_param, const int size_param, const unsigned char* mem_model, const int size_model, const bool use_vulkan)
{
    if (use_vulkan && ncnn::get_gpu_count() == 0)
    {
        // no gpu
        std::cout << "[fastestdet]Err Your GPU count is Zero" << std::endl;
        return NULL;
    }

    _fastestdet* fastestdetNet = new _fastestdet;

    fastestdetNet->net.opt.use_vulkan_compute = use_vulkan;
    fastestdetNet->net.opt.num_threads = ncnn::get_big_cpu_count();

    fastestdetNet->param.clear();
    fastestdetNet->model.clear();

    fastestdetNet->param.insert(fastestdetNet->param.end(), mem_param, mem_param + size_param);
    fastestdetNet->model.insert(fastestdetNet->model.end(), mem_model, mem_model + size_model);

    fastestdetNet->param.push_back(0);

    if (fastestdetNet->net.load_param_mem((char*)fastestdetNet->param.data()) != 0)
    {
        std::cout << "[fastestdet]Err Read Param Failed" << std::endl;
        delete fastestdetNet;
        return NULL;
    }
    if (fastestdetNet->net.load_model(fastestdetNet->model.data()) == 0)
    {
        std::cout << "[fastestdet]Err Read Model Failed" << std::endl;
        delete fastestdetNet;
        return NULL;
    }
    return fastestdetNet;
}

extern "C" int __declspec(dllexport) __stdcall fastestdet_Deal(__fastestdet fastestdet, const unsigned char* img_src, const int img_size, const int target_size, const float prob_threshold, const float nms_threshold,const int class_num, Object * *ResList)
{
    if (fastestdet == NULL)
    {
        std::cout << "[fastestdet]Not Init" << std::endl;
        return 0;
    }

    cv::_InputArray pic_arr(img_src, img_size);
    cv::Mat src_mat = cv::imdecode(pic_arr, cv::IMREAD_UNCHANGED);

    if (src_mat.empty())
    {
        std::cout << "[fastestdet]ERR Cant Read Img" << std::endl;
        return 0;
    }

    /*
        unsigned char *mat_data;
        if (!BMP24TOMAT(img_src, &mat_data))
        {
            std::cout << "[Yolov7]Err Read BMP Failed" << endl;
            return 0;
        }
        int img_w, img_h;
        GetBMPSize(img_src, img_w, img_h);
    */

    int img_width = src_mat.cols;
    int img_height = src_mat.rows;

    // resize of input image data
    ncnn::Mat input = ncnn::Mat::from_pixels_resize(src_mat.data, ncnn::Mat::PIXEL_BGR, \
        src_mat.cols, src_mat.rows, target_size, target_size);
    // Normalization of input image data
    const float mean_vals[3] = { 0.f, 0.f, 0.f };
    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    input.substract_mean_normalize(mean_vals, norm_vals);

    // creat extractor
    ncnn::Extractor ex = fastestdet->net.create_extractor();
    ex.set_num_threads(1);

    //set input tensor
    ex.input("input.1", input);

    // get output tensor
    ncnn::Mat output;
    ex.extract("758", output);

    // handle output tensor
    std::vector<TargetBox> target_boxes;

    for (int h = 0; h < output.h; h++)
    {
        for (int w = 0; w < output.h; w++)
        {
            // 前景概率
            int obj_score_index = (0 * output.h * output.w) + (h * output.w) + w;
            float obj_score = output[obj_score_index];

            // 解析类别
            int category;
            float max_score = 0.0f;
            for (size_t i = 0; i < class_num; i++)
            {
                int obj_score_index = ((5 + i) * output.h * output.w) + (h * output.w) + w;
                float cls_score = output[obj_score_index];
                if (cls_score > max_score)
                {
                    max_score = cls_score;
                    category = i;
                }
            }
            float score = pow(max_score, 0.4) * pow(obj_score, 0.6);

            // 阈值筛选
            if (score > prob_threshold)
            {
                // 解析坐标
                int x_offset_index = (1 * output.h * output.w) + (h * output.w) + w;
                int y_offset_index = (2 * output.h * output.w) + (h * output.w) + w;
                int box_width_index = (3 * output.h * output.w) + (h * output.w) + w;
                int box_height_index = (4 * output.h * output.w) + (h * output.w) + w;

                float x_offset = Tanh(output[x_offset_index]);
                float y_offset = Tanh(output[y_offset_index]);
                float box_width = Sigmoid(output[box_width_index]);
                float box_height = Sigmoid(output[box_height_index]);

                float cx = (w + x_offset) / output.w;
                float cy = (h + y_offset) / output.h;

                int x1 = (int)((cx - box_width * 0.5) * img_width);
                int y1 = (int)((cy - box_height * 0.5) * img_height);
                int x2 = (int)((cx + box_width * 0.5) * img_width);
                int y2 = (int)((cy + box_height * 0.5) * img_height);

                target_boxes.push_back(TargetBox{ x1, y1, x2, y2, category, score });
            }
        }
    }

    // NMS处理
    std::vector<TargetBox> nms_boxes;
    nmsHandle(target_boxes, nms_boxes, nms_threshold);

    if (nms_boxes.size() == 0)
        return 0;

    *ResList = new Object[nms_boxes.size()];
    // std::cout << "[DEBUG] " << proposals.size() * sizeof(Object) <<' '<<_msize(*ResList) << std::endl;
    for (int i = 0; i < nms_boxes.size(); i++)
    {
        (*ResList)[i].label = nms_boxes[i].category;
        (*ResList)[i].prob = nms_boxes[i].score;
        (*ResList)[i].rect.x = nms_boxes[i].x1;
        (*ResList)[i].rect.y = nms_boxes[i].y1;
        (*ResList)[i].rect.width = nms_boxes[i].x2 - nms_boxes[i].x1;
        (*ResList)[i].rect.height = nms_boxes[i].y2 - nms_boxes[i].y1;
    }
    return nms_boxes.size();
}

extern "C" void __declspec(dllexport) __stdcall fastestdet_DestructRet(Object * ResList)
{
    delete[] ResList;
}

extern "C" void __declspec(dllexport) __stdcall fastestdet_Destroy(__fastestdet fastestdet)
{
    fastestdet->net.clear();
    delete fastestdet;
}