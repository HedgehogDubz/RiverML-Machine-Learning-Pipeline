// Machine Learning Visualizer, desktop version using raylib.
// Recreates the TypeScript web version: network view on top, data view below,
// sidebar with controls on the left.
#include "raylib.h"
#include "neuralnetwork.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

//////////////////// app state ////////////////////

enum class NetFormat { Val1in1Out, Val2in1Out, Cat2in2Out };
enum class DataFormat { TestVsOutput, Output, Error, Test, None };
enum class NetView { Best, All };
enum class TrainMethod { Genetic, Backprop };

static const char* kFormatNames[] = { "Value: 1in1out", "Value: 2in1out", "Cat: 2in2out" };
static const char* kDataNames2In[] = { "Output", "Error", "Test", "None" };
static const char* kDataNames1In[] = { "Test vs Output", "Output", "Error", "Test", "None" };
static const char* kViewNames[] = { "Best", "All" };
static const char* kMethodNames[] = { "Genetic", "Backprop" };

static const char* kTest1Names[] = { "Sine", "Square", "Sawtooth", "Triangle", "Abs", "Cubic",
                                     "Polynomial", "Step", "Gaussian", "Tanh", "Sinc", "Noisy Sine" };
static const char* kTest2Names[] = { "Wave", "Radial", "XY", "Checkerboard", "Spiral", "Diagonal",
                                     "Gaussian", "Saddle", "Ripple", "Peaks", "Step 2D", "Swiss Roll" };
static const char* kTestCNames[] = { "Circle", "Square", "Quadrants", "Donut", "XOR", "Diagonal",
                                     "Stripes", "Checkerboard", "Spiral", "Moons", "Diamond", "Cross" };

static NetFormat gFormat = NetFormat::Val2in1Out;
static int gDataIdx = 0;      // index into the active data format list
static NetView gView = NetView::Best;
static TrainMethod gMethod = TrainMethod::Genetic;
static int gTestFnIdx = 0;
static bool gStarted = false;
static float gGensPerDraw = 1;
static float gLearningRate = 0.01f;
static float gMomentum = 0.9f;
static float gInput1 = 0;
static float gInput2 = 0;

static int gNumWeightsMut = 100;
static int gNumBiasesMut = 100;
static double gWeightStrength = 0.01;
static double gBiasStrength = 0.01;
static double gLastError = 1e300;

static Population gPop;
static int gInputSize = 2;
static int gOutputSize = 1;
static std::vector<int> gHidden = { 7, 10, 20, 20, 10, 7 };

static std::string gStatusMsg;
static double gStatusUntil = 0;

// class colors for Cat 2in2out
static const Color kColor1 = { 100, 150, 255, 255 };
static const Color kColor2 = { 255, 100, 100, 255 };
static const Color kAccent = { 102, 110, 234, 255 };

//////////////////// test functions ////////////////////

static void testFn(const double* in, double* out) {
    double x = in[0];
    if (gFormat == NetFormat::Val1in1Out) {
        switch (gTestFnIdx) {
            case 0: out[0] = sin(x * PI); break;
            case 1: out[0] = sin(x * PI * 2) > 0 ? 1 : -1; break;
            case 2: out[0] = fmod(x + 1.0, 0.5) * 4 - 1; break;
            case 3: out[0] = 1 - 4 * fabs(round(x * 0.5) - x * 0.5); break;
            case 4: out[0] = fabs(x) * 2 - 1; break;
            case 5: out[0] = fmax(-1.0, fmin(1.0, x * x * x * 4)); break;
            case 6: out[0] = fmax(-1.0, fmin(1.0, 2 * x * x * x - 3 * x * x + x + 0.5)); break;
            case 7: out[0] = x < -0.5 ? -1 : x < 0 ? -0.3 : x < 0.5 ? 0.3 : 1; break;
            case 8: out[0] = exp(-x * x * 5) * 2 - 1; break;
            case 9: out[0] = tanh(x * 3); break;
            case 10: out[0] = x == 0 ? 1 : sin(x * 6) / (x * 6); break;
            default: out[0] = sin(x * PI) * 0.7 + sin(x * 7) * 0.3; break;
        }
        return;
    }
    double y = in[1];
    if (gFormat == NetFormat::Val2in1Out) {
        switch (gTestFnIdx) {
            case 0: out[0] = x > sin(y * 2 * PI) ? 1 : -1; break;
            case 1: out[0] = fmax(fmin(1 - 2 * (x * x + y * y), 1.0), -1.0); break;
            case 2: out[0] = x * y; break;
            case 3: out[0] = ((int)floor(x * 4) % 2 + 2) % 2 == ((int)floor(y * 4) % 2 + 2) % 2 ? 1 : -1; break;
            case 4: { double a = atan2(y, x), r = sqrt(x * x + y * y);
                      out[0] = sin(a * 3 + r * 5) > 0 ? 1 : -1; break; }
            case 5: out[0] = x + y > 0 ? 1 : -1; break;
            case 6: out[0] = exp(-(x * x + y * y) * 3) * 2 - 1; break;
            case 7: out[0] = fmax(-1.0, fmin(1.0, x * x - y * y)); break;
            case 8: { double d = sqrt(x * x + y * y); out[0] = sin(d * 8) * exp(-d * 2); break; }
            case 9: out[0] = fmax(-1.0, fmin(1.0, exp(-((x - 0.5) * (x - 0.5) + y * y) * 5)
                                                 - exp(-((x + 0.5) * (x + 0.5) + y * y) * 5))); break;
            case 10: out[0] = x > 0.3 ? 1 : x < -0.3 ? -1 : y > 0 ? 0.5 : -0.5; break;
            default: out[0] = sin(x * 3 + y * 2) * cos(x * 2 - y * 3) > 0 ? 1 : -1; break;
        }
        return;
    }
    // Cat 2in2out, one hot classes
    bool c1;
    switch (gTestFnIdx) {
        case 0: c1 = x * x + y * y < 0.5; break;
        case 1: c1 = fabs(x) < 0.5 && fabs(y) < 0.5; break;
        case 2: c1 = x * y > 0; break;
        case 3: { double r = x * x + y * y; c1 = r > 0.25 && r < 0.75; break; }
        case 4: c1 = (x > 0) != (y > 0); break;
        case 5: c1 = x > y; break;
        case 6: c1 = sin(x * 6) > 0; break;
        case 7: c1 = ((int)(floor((x + 1) * 3) + floor((y + 1) * 3)) % 2) == 0; break;
        case 8: { double a = atan2(y, x), r = sqrt(x * x + y * y);
                  c1 = sin(a * 2 + r * 6) > 0; break; }
        case 9: c1 = x * x + (y - 0.3) * (y - 0.3) < 0.6 && x * x + (y + 0.3) * (y + 0.3) > 0.3; break;
        case 10: c1 = fabs(x) + fabs(y) < 0.7; break;
        default: c1 = fabs(x) < 0.2 || fabs(y) < 0.2; break;
    }
    out[0] = c1 ? 1 : 0;
    out[1] = c1 ? 0 : 1;
}

static std::vector<double> testVec(const std::vector<double>& in) {
    std::vector<double> out(gOutputSize, 0.0);
    testFn(in.data(), out.data());
    return out;
}

//////////////////// setup ////////////////////

static void createTrials() {
    gPop.trialInputs.clear();
    gPop.trialOutputs.clear();
    gPop.testInputs.clear();
    gPop.testOutputs.clear();
    if (gInputSize == 1) {
        for (int i = 0; i <= 20; i++) {
            gPop.trialInputs.push_back({ -1.0 + i * 0.1 });
        }
        for (int i = 0; i < 20; i++) {
            gPop.testInputs.push_back({ -0.95 + i * 0.1 });
        }
    } else {
        for (int i = 0; i <= 20; i++)
            for (int j = 0; j <= 20; j++)
                gPop.trialInputs.push_back({ -1.0 + i * 0.1, -1.0 + j * 0.1 });
        for (int i = 0; i < 20; i++)
            for (int j = 0; j < 20; j++)
                gPop.testInputs.push_back({ -0.95 + i * 0.1, -0.95 + j * 0.1 });
    }
    for (auto& in : gPop.trialInputs) gPop.trialOutputs.push_back(testVec(in));
    for (auto& in : gPop.testInputs) gPop.testOutputs.push_back(testVec(in));
}

static void rebuildPopulation() {
    gInputSize = gFormat == NetFormat::Val1in1Out ? 1 : 2;
    gOutputSize = gFormat == NetFormat::Cat2in2Out ? 2 : 1;
    Activation outAct = gFormat == NetFormat::Cat2in2Out ? Activation::Sigmoid : Activation::Tanh;
    int count = gMethod == TrainMethod::Genetic ? 16 : 1;
    gPop.init(count, gInputSize, gHidden, gOutputSize, Activation::ReLU, outAct);
    createTrials();
    gWeightStrength = 0.01;
    gBiasStrength = 0.01;
    gLastError = 1e300;
}

static void showStatus(const std::string& msg) {
    gStatusMsg = msg;
    gStatusUntil = GetTime() + 4.0;
}

//////////////////// color helpers ////////////////////

static Color numToColorBlack(double v) {
    double t = tanh(v);
    if (v >= 0) return { 0, (unsigned char)(t * 255), 0, 255 };
    return { (unsigned char)(-t * 255), 0, 0, 255 };
}

static Color numToColorWhite(double v) {
    double t = tanh(v);
    if (v >= 0) {
        unsigned char c = (unsigned char)((1 - t) * 255);
        return { c, 255, c, 255 };
    }
    unsigned char c = (unsigned char)((1 + t) * 255);
    return { 255, c, c, 255 };
}

static Color lerpColor(Color a, Color b, double w) {
    w = fmax(0.0, fmin(1.0, w));
    return { (unsigned char)(a.r * w + b.r * (1 - w)),
             (unsigned char)(a.g * w + b.g * (1 - w)),
             (unsigned char)(a.b * w + b.b * (1 - w)), 255 };
}

//////////////////// simple immediate mode UI ////////////////////

static int gActiveSlider = -1;

static bool uiButton(Rectangle r, const char* text) {
    bool hover = CheckCollisionPointRec(GetMousePosition(), r);
    Color bg = hover ? Color{ 85, 104, 211, 255 } : kAccent;
    DrawRectangleRec(r, bg);
    int fs = 14;
    int tw = MeasureText(text, fs);
    DrawText(text, (int)(r.x + r.width / 2 - tw / 2), (int)(r.y + r.height / 2 - fs / 2), fs, WHITE);
    return hover && IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
}

static bool uiCycler(Rectangle r, const char* const* options, int count, int* index) {
    DrawRectangleRec(r, WHITE);
    DrawRectangleLinesEx(r, 1, Color{ 200, 200, 200, 255 });
    Rectangle leftBtn = { r.x, r.y, 22, r.height };
    Rectangle rightBtn = { r.x + r.width - 22, r.y, 22, r.height };
    Vector2 m = GetMousePosition();
    bool changed = false;
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        if (CheckCollisionPointRec(m, leftBtn)) { *index = (*index + count - 1) % count; changed = true; }
        else if (CheckCollisionPointRec(m, rightBtn)) { *index = (*index + 1) % count; changed = true; }
    }
    int fs = 12;
    DrawText("<", (int)(leftBtn.x + 8), (int)(r.y + r.height / 2 - fs / 2), fs, DARKGRAY);
    DrawText(">", (int)(rightBtn.x + 8), (int)(r.y + r.height / 2 - fs / 2), fs, DARKGRAY);
    const char* text = options[*index];
    int tw = MeasureText(text, fs);
    DrawText(text, (int)(r.x + r.width / 2 - tw / 2), (int)(r.y + r.height / 2 - fs / 2), fs, BLACK);
    return changed;
}

static float uiSlider(Rectangle r, float minV, float maxV, float value, int id) {
    Vector2 m = GetMousePosition();
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && CheckCollisionPointRec(m, r)) gActiveSlider = id;
    if (!IsMouseButtonDown(MOUSE_BUTTON_LEFT) && gActiveSlider == id) gActiveSlider = -1;
    if (gActiveSlider == id) {
        float t = (m.x - r.x) / r.width;
        value = minV + fmaxf(0.0f, fminf(1.0f, t)) * (maxV - minV);
    }
    float cy = r.y + r.height / 2;
    DrawLineEx({ r.x, cy }, { r.x + r.width, cy }, 3, Color{ 200, 200, 200, 255 });
    float kx = r.x + (value - minV) / (maxV - minV) * r.width;
    DrawCircle((int)kx, (int)cy, 7, kAccent);
    return value;
}

//////////////////// network drawing ////////////////////

static void drawValueText(double v, int cx, int cy, double maxWidth) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.3f", floor(v * 1000) / 1000);
    // trim trailing chars until the text fits
    int len = (int)strlen(buf);
    while (len > 0) {
        if (buf[len - 1] == '.') { buf[--len] = 0; continue; }
        int w = MeasureText(buf, 10);
        if (w < maxWidth) {
            DrawText(buf, cx - w / 2, cy - 5, 10, BLACK);
            return;
        }
        buf[--len] = 0;
    }
}

static void drawNetwork(NeuralNetwork& nn, Rectangle rect, bool showError) {
    int L = nn.numLayers();
    float spaceX = rect.width / (L + 1);
    float bottomPad = showError ? 30.0f : 0.0f;

    // weight lines
    for (int l = 1; l < L; l++) {
        float x = rect.x + (l + 1) * spaceX;
        float lastX = rect.x + l * spaceX;
        float spaceY = (rect.height - bottomPad) / (nn.sizes[l] + 1);
        float lastSpaceY = (rect.height - bottomPad) / (nn.sizes[l - 1] + 1);
        for (int n = 0; n < nn.sizes[l]; n++) {
            float y = rect.y + (n + 1) * spaceY;
            const double* w = &nn.layers[l].weights[(size_t)n * nn.layers[l].prevSize];
            for (int p = 0; p < nn.sizes[l - 1]; p++) {
                float ly = rect.y + (p + 1) * lastSpaceY;
                DrawLineEx({ x, y }, { lastX, ly }, 2, numToColorBlack(w[p]));
            }
        }
    }

    // neurons
    for (int l = 0; l < L; l++) {
        float x = rect.x + (l + 1) * spaceX;
        float spaceY = (rect.height - bottomPad) / (nn.sizes[l] + 1);
        float r1 = fminf(spaceX, spaceY) / 2.1f;
        float r2 = fminf(spaceX, spaceY) / 2.3f;
        for (int n = 0; n < nn.sizes[l]; n++) {
            float y = rect.y + (n + 1) * spaceY;
            double bias = l == 0 ? 0 : nn.layers[l].biases[n];
            DrawCircleV({ x, y }, r1, numToColorBlack(bias));
            DrawCircleV({ x, y }, r2, numToColorWhite(nn.layers[l].values[n]));
            drawValueText(nn.layers[l].values[n], (int)x, (int)y, r2 * 1.8);
        }
    }

    if (showError) {
        char buf[96];
        snprintf(buf, sizeof(buf), "e RMSE: %.5f", nn.error);
        int w = MeasureText(buf, 10);
        DrawText(buf, (int)(rect.x + rect.width / 2 - w / 2), (int)(rect.y + rect.height - 26), 10, BLACK);
        snprintf(buf, sizeof(buf), "u MAE: %.5f", nn.meanError);
        w = MeasureText(buf, 10);
        DrawText(buf, (int)(rect.x + rect.width / 2 - w / 2), (int)(rect.y + rect.height - 13), 10, BLACK);
    }
}

//////////////////// data drawing ////////////////////

// grid for 2 input formats, cellColor decides the fill, cellText optional
template <typename ColorFn, typename TextFn>
static void drawGrid(Rectangle rect, int rows, int cols, bool showText, bool showHeaders,
                     ColorFn cellColor, TextFn cellText) {
    int headerOffset = showHeaders ? 1 : 0;
    float spaceX = rect.width / (cols + headerOffset);
    float spaceY = rect.height / (rows + headerOffset);
    DrawRectangleRec(rect, Color{ 224, 224, 224, 255 });
    int fs = (int)fmaxf(8.0f, fminf(spaceX, spaceY) * 0.3f);

    if (showHeaders) {
        char buf[16];
        for (int i = 0; i < cols; i++) {
            double v = -1.0 + i * 2.0 / (cols - 1);
            snprintf(buf, sizeof(buf), "%.1f", v);
            int tw = MeasureText(buf, fs);
            DrawText(buf, (int)(rect.x + (i + 1) * spaceX + spaceX / 2 - tw / 2),
                     (int)(rect.y + spaceY / 2 - fs / 2), fs, BLACK);
        }
        for (int j = 0; j < rows; j++) {
            double v = -1.0 + j * 2.0 / (rows - 1);
            snprintf(buf, sizeof(buf), "%.1f", v);
            int tw = MeasureText(buf, fs);
            DrawText(buf, (int)(rect.x + spaceX / 2 - tw / 2),
                     (int)(rect.y + (j + 1) * spaceY + spaceY / 2 - fs / 2), fs, BLACK);
        }
        DrawRectangle((int)rect.x, (int)rect.y, (int)spaceX, (int)spaceY, Color{ 192, 192, 192, 255 });
    }

    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            float x = rect.x + (i + headerOffset) * spaceX;
            float y = rect.y + (j + headerOffset) * spaceY;
            double in1 = -1.0 + i * 2.0 / (cols - 1);
            double in2 = -1.0 + j * 2.0 / (rows - 1);
            DrawRectangle((int)x, (int)y, (int)ceilf(spaceX) + (showText ? -1 : 1),
                          (int)ceilf(spaceY) + (showText ? -1 : 1), cellColor(in1, in2));
            if (showText) {
                char buf[32];
                cellText(in1, in2, buf, sizeof(buf));
                int tw = MeasureText(buf, fs);
                DrawText(buf, (int)(x + spaceX / 2 - tw / 2), (int)(y + spaceY / 2 - fs / 2), fs, BLACK);
            }
        }
    }
}

// line graph for the 1 input format
static void drawLineGraph(Rectangle rect, std::vector<std::pair<double (*)(double), Color>>& lines) {
    DrawRectangleRec(rect, Color{ 248, 248, 248, 255 });
    double yLow = -1.2, yHigh = 1.2;
    auto sx = [&](double x) { return rect.x + (float)((x + 1) / 2 * rect.width); };
    auto sy = [&](double y) { return rect.y + rect.height - (float)((y - yLow) / (yHigh - yLow) * rect.height); };

    // grid lines
    for (double v = -1.0; v <= 1.001; v += 0.2) {
        DrawLineEx({ sx(v), rect.y }, { sx(v), rect.y + rect.height }, 1, Color{ 221, 221, 221, 255 });
    }
    for (double v = -1.2; v <= 1.201; v += 0.2) {
        DrawLineEx({ rect.x, sy(v) }, { rect.x + rect.width, sy(v) }, 1, Color{ 221, 221, 221, 255 });
    }
    // axes
    DrawLineEx({ rect.x, sy(0) }, { rect.x + rect.width, sy(0) }, 1, Color{ 153, 153, 153, 255 });
    DrawLineEx({ sx(0), rect.y }, { sx(0), rect.y + rect.height }, 1, Color{ 153, 153, 153, 255 });
    // labels
    char buf[16];
    for (double v = -1.0; v <= 1.001; v += 0.5) {
        snprintf(buf, sizeof(buf), "%.1f", v);
        DrawText(buf, (int)(sx(v) - MeasureText(buf, 10) / 2), (int)(rect.y + rect.height + 2), 10, GRAY);
    }
    for (double v = -1.0; v <= 1.001; v += 0.5) {
        snprintf(buf, sizeof(buf), "%.1f", v);
        DrawText(buf, (int)(rect.x - MeasureText(buf, 10) - 3), (int)(sy(v) - 5), 10, GRAY);
    }

    int numPoints = 200;
    for (auto& line : lines) {
        double prevY = 0;
        for (int i = 0; i <= numPoints; i++) {
            double x = -1.0 + (i / (double)numPoints) * 2.0;
            double y = fmax(yLow, fmin(yHigh, line.first(x)));
            if (i > 0) {
                double px = -1.0 + ((i - 1) / (double)numPoints) * 2.0;
                DrawLineEx({ sx(px), sy(prevY) }, { sx(x), sy(y) }, 2, line.second);
            }
            prevY = y;
        }
    }

    // dashed marker at input 1
    float mx = sx(gInput1);
    for (float y = rect.y; y < rect.y + rect.height; y += 8) {
        DrawLineEx({ mx, y }, { mx, fminf(y + 4, rect.y + rect.height) }, 1, Color{ 0, 0, 0, 100 });
    }
}

// helper for line graph function pointers
static double predictLine(double x) {
    std::vector<double> in = { x };
    return gPop.nets[0].run(in)[0];
}
static double testLine(double x) {
    double in[2] = { x, 0 }, out[2];
    testFn(in, out);
    return out[0];
}
static double errorLine(double x) { return predictLine(x) - testLine(x); }

//////////////////// seed buttons ////////////////////

static void copySeed() {
    gPop.sortByError();
    std::string seed = gPop.nets[0].toSeed();
    SetClipboardText(seed.c_str());
    FILE* f = fopen("seed.txt", "w");
    if (f) { fputs(seed.c_str(), f); fclose(f); }
    showStatus(TextFormat("Seed copied to clipboard and seed.txt (%d chars)", (int)seed.size()));
}

static void loadSeed() {
    const char* clip = GetClipboardText();
    if (!clip || !*clip) { showStatus("Clipboard is empty"); return; }
    NeuralNetwork nn;
    if (!NeuralNetwork::fromSeed(clip, nn)) { showStatus("Invalid seed in clipboard"); return; }
    if (nn.inputSize() != gInputSize || nn.outputSize() != gOutputSize) {
        showStatus(TextFormat("Seed is %din %dout, switch network format first",
                              nn.inputSize(), nn.outputSize()));
        return;
    }
    gHidden.assign(nn.sizes.begin() + 1, nn.sizes.end() - 1);
    gPop.setAll(nn);
    showStatus("Seed loaded");
}

//////////////////// main ////////////////////

int main() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(1280, 760, "Machine Learning Visualizer");
    SetTargetFPS(60);

    rebuildPopulation();

    while (!WindowShouldClose()) {
        // training step
        if (gStarted) {
            int gens = (int)gGensPerDraw;
            for (int g = 0; g < gens; g++) {
                double best;
                if (gMethod == TrainMethod::Genetic) {
                    best = gPop.runGeneration(gNumWeightsMut, gWeightStrength, gNumBiasesMut, gBiasStrength);
                    // adapt mutation strength like the web version
                    if (best == gLastError) {
                        gWeightStrength = fmax(0.0001, gWeightStrength / 1.001);
                        gBiasStrength = fmax(0.0001, gBiasStrength / 1.001);
                    } else {
                        gWeightStrength = fmin(1.0, gWeightStrength * 1.001);
                        gBiasStrength = fmin(1.0, gBiasStrength * 1.001);
                    }
                } else {
                    best = gPop.trainBackprop(gLearningRate, gMomentum);
                }
                gLastError = best;
            }
            gPop.computeTestErr();
        }

        BeginDrawing();
        ClearBackground(Color{ 248, 249, 250, 255 });

        float W = (float)GetScreenWidth();
        float H = (float)GetScreenHeight();
        float sidebarW = 260;
        float canvasX = sidebarW + 10;
        float canvasW = W - canvasX - 10;
        float headerH = 30;
        float halfH = (H - headerH) / 2;

        //////////////////// sidebar ////////////////////
        DrawRectangle(0, 0, (int)sidebarW, (int)H, Color{ 242, 242, 247, 255 });
        DrawText("Controls", 15, 12, 20, kAccent);
        float y = 45;
        float cw = sidebarW - 30;
        char lbl[64];

        DrawText("Network Format", 15, (int)y, 10, GRAY); y += 13;
        {
            int idx = (int)gFormat;
            if (uiCycler({ 15, y, cw, 24 }, kFormatNames, 3, &idx)) {
                gFormat = (NetFormat)idx;
                gTestFnIdx = 0;
                gDataIdx = 0;
                gStarted = false;
                rebuildPopulation();
            }
            y += 32;
        }

        DrawText("Test Function", 15, (int)y, 10, GRAY); y += 13;
        {
            const char* const* names = gFormat == NetFormat::Val1in1Out ? kTest1Names
                                     : gFormat == NetFormat::Val2in1Out ? kTest2Names : kTestCNames;
            if (uiCycler({ 15, y, cw, 24 }, names, 12, &gTestFnIdx)) {
                createTrials();
            }
            y += 32;
        }

        DrawText("Data Format", 15, (int)y, 10, GRAY); y += 13;
        {
            bool is1in = gFormat == NetFormat::Val1in1Out;
            uiCycler({ 15, y, cw, 24 }, is1in ? kDataNames1In : kDataNames2In, is1in ? 5 : 4, &gDataIdx);
            y += 32;
        }

        DrawText("Network View", 15, (int)y, 10, GRAY); y += 13;
        {
            int idx = (int)gView;
            uiCycler({ 15, y, cw, 24 }, kViewNames, 2, &idx);
            gView = (NetView)idx;
            y += 32;
        }

        DrawText("Training Method", 15, (int)y, 10, GRAY); y += 13;
        {
            int idx = (int)gMethod;
            if (uiCycler({ 15, y, cw, 24 }, kMethodNames, 2, &idx)) {
                TrainMethod newMethod = (TrainMethod)idx;
                // keep the best network when switching methods
                for (auto& nn : gPop.nets) gPop.measure(nn);
                gPop.sortByError();
                NeuralNetwork best = gPop.nets[0];
                gMethod = newMethod;
                int count = gMethod == TrainMethod::Genetic ? 16 : 1;
                gPop.nets.assign(count, best);
                gPop.targetCount = count;
                gPop.hasWorker = false;
                gPop.workerBestError = 1e300;
            }
            y += 32;
        }

        if (gMethod == TrainMethod::Backprop) {
            snprintf(lbl, sizeof(lbl), "Learning Rate: %.3f", gLearningRate);
            DrawText(lbl, 15, (int)y, 10, GRAY); y += 13;
            gLearningRate = uiSlider({ 15, y, cw, 18 }, 0.001f, 0.5f, gLearningRate, 1);
            y += 26;
            snprintf(lbl, sizeof(lbl), "Momentum: %.2f", gMomentum);
            DrawText(lbl, 15, (int)y, 10, GRAY); y += 13;
            gMomentum = uiSlider({ 15, y, cw, 18 }, 0.0f, 0.99f, gMomentum, 2);
            y += 26;
        }

        snprintf(lbl, sizeof(lbl), "Gens/Draw: %d", (int)gGensPerDraw);
        DrawText(lbl, 15, (int)y, 10, GRAY); y += 13;
        gGensPerDraw = uiSlider({ 15, y, cw, 18 }, 1, 100, gGensPerDraw, 3);
        y += 26;

        snprintf(lbl, sizeof(lbl), "Input 1: %.2f", gInput1);
        DrawText(lbl, 15, (int)y, 10, GRAY); y += 13;
        gInput1 = uiSlider({ 15, y, cw, 18 }, -1, 1, gInput1, 4);
        y += 26;

        if (gInputSize >= 2) {
            snprintf(lbl, sizeof(lbl), "Input 2: %.2f", gInput2);
            DrawText(lbl, 15, (int)y, 10, GRAY); y += 13;
            gInput2 = uiSlider({ 15, y, cw, 18 }, -1, 1, gInput2, 5);
            y += 26;
        }

        y += 8;
        if (uiButton({ 15, y, cw / 2 - 4, 28 }, gStarted ? "Stop" : "Start")) gStarted = !gStarted;
        if (uiButton({ 15 + cw / 2 + 4, y, cw / 2 - 4, 28 }, "Reset")) rebuildPopulation();
        y += 36;
        if (uiButton({ 15, y, cw / 2 - 4, 28 }, "Copy Seed")) copySeed();
        if (uiButton({ 15 + cw / 2 + 4, y, cw / 2 - 4, 28 }, "Load Seed")) loadSeed();
        y += 36;

        if (GetTime() < gStatusUntil) {
            DrawText(gStatusMsg.c_str(), 15, (int)y, 10, DARKGRAY);
        }

        //////////////////// header ////////////////////
        DrawRectangle((int)canvasX, 0, (int)canvasW, (int)headerH, Color{ 224, 224, 224, 255 });
        char header[256];
        snprintf(header, sizeof(header),
                 "Gen: %d | Train RMSE: %.4f MAE: %.4f | Test RMSE: %.4f MAE: %.4f | FPS: %d",
                 gPop.generation, gPop.trainRMSE, gPop.trainMAE, gPop.testRMSE, gPop.testMAE, GetFPS());
        DrawText(header, (int)canvasX + 5, (int)(headerH / 2 - 6), 12, BLACK);

        //////////////////// network view (top half) ////////////////////
        std::vector<double> currentInputs = gInputSize == 1
            ? std::vector<double>{ (double)gInput1 }
            : std::vector<double>{ (double)gInput1, (double)gInput2 };

        Rectangle netRect = { canvasX, headerH, canvasW, halfH };
        if (gView == NetView::Best || gPop.nets.size() == 1) {
            NeuralNetwork shown = gPop.nets[0];
            shown.run(currentInputs);
            drawNetwork(shown, netRect, false);
        } else {
            int cols = 4;
            int rows = ((int)gPop.nets.size() + cols - 1) / cols;
            float cw2 = netRect.width / cols;
            float ch2 = netRect.height / rows;
            for (size_t i = 0; i < gPop.nets.size(); i++) {
                NeuralNetwork shown = gPop.nets[i];
                shown.run(currentInputs);
                Rectangle cell = { netRect.x + (i % cols) * cw2, netRect.y + (i / cols) * ch2, cw2, ch2 };
                drawNetwork(shown, cell, true);
            }
        }

        // divider
        DrawRectangle((int)canvasX, (int)(headerH + halfH - 1), (int)canvasW, 3, BLACK);

        //////////////////// data view (bottom half) ////////////////////
        Rectangle dataRect = { canvasX, headerH + halfH + 2, canvasW, halfH - 2 };
        NeuralNetwork& net = gPop.nets[0];

        if (gFormat == NetFormat::Val1in1Out) {
            Rectangle graph = { dataRect.x + 30, dataRect.y + 8, dataRect.width - 45, dataRect.height - 24 };
            std::vector<std::pair<double (*)(double), Color>> lines;
            DataFormat df = (DataFormat)gDataIdx;
            if (df == DataFormat::TestVsOutput) {
                lines.push_back({ testLine, Color{ 76, 175, 80, 255 } });
                lines.push_back({ predictLine, Color{ 33, 150, 243, 255 } });
            } else if (df == DataFormat::Output) {
                lines.push_back({ predictLine, Color{ 33, 150, 243, 255 } });
            } else if (df == DataFormat::Error) {
                lines.push_back({ errorLine, Color{ 229, 57, 53, 255 } });
            } else if (df == DataFormat::Test) {
                lines.push_back({ testLine, Color{ 76, 175, 80, 255 } });
            }
            drawLineGraph(graph, lines);
        } else {
            // two input formats: fine grid on the left, 11x11 with values on the right
            float gap = 3;
            Rectangle leftR = { dataRect.x, dataRect.y, dataRect.width / 2 - gap, dataRect.height };
            Rectangle rightR = { dataRect.x + dataRect.width / 2 + gap, dataRect.y,
                                 dataRect.width / 2 - gap, dataRect.height };
            int dfShifted = gDataIdx; // 0 output, 1 error, 2 test, 3 none
            bool isCat = gFormat == NetFormat::Cat2in2Out;

            auto predCell = [&](double in1, double in2) {
                std::vector<double> in = { in1, in2 };
                return net.run(in);
            };
            auto colorForValue = [&](double in1, double in2) -> Color {
                if (isCat) return lerpColor(kColor1, kColor2, predCell(in1, in2)[0]);
                return numToColorWhite(predCell(in1, in2)[0]);
            };
            auto colorForTest = [&](double in1, double in2) -> Color {
                double in[2] = { in1, in2 }, out[2];
                testFn(in, out);
                if (isCat) return lerpColor(kColor1, kColor2, out[0]);
                return numToColorWhite(out[0]);
            };
            auto colorForError = [&](double in1, double in2) -> Color {
                double in[2] = { in1, in2 }, out[2];
                testFn(in, out);
                std::vector<double> pred = predCell(in1, in2);
                if (isCat) {
                    double err = (fabs(pred[0] - out[0]) + fabs(pred[1] - out[1])) / 2;
                    return lerpColor(Color{ 255, 255, 255, 255 }, Color{ 255, 0, 0, 255 }, 1 - err);
                }
                return numToColorWhite(pred[0] - out[0]);
            };
            auto textForValue = [&](double in1, double in2, char* buf, size_t n) {
                std::vector<double> pred = predCell(in1, in2);
                if (isCat) snprintf(buf, n, "%.1f,%.1f", pred[0], pred[1]);
                else snprintf(buf, n, "%.2f", pred[0]);
            };
            auto textForTest = [&](double in1, double in2, char* buf, size_t n) {
                double in[2] = { in1, in2 }, out[2];
                testFn(in, out);
                if (isCat) snprintf(buf, n, "%.1f,%.1f", out[0], out[1]);
                else snprintf(buf, n, "%.2f", out[0]);
            };
            auto textForError = [&](double in1, double in2, char* buf, size_t n) {
                double in[2] = { in1, in2 }, out[2];
                testFn(in, out);
                std::vector<double> pred = predCell(in1, in2);
                if (isCat) snprintf(buf, n, "%.1f", (fabs(pred[0] - out[0]) + fabs(pred[1] - out[1])) / 2);
                else snprintf(buf, n, "%.2f", pred[0] - out[0]);
            };

            int fine = 101;
            switch (dfShifted) {
                case 0: // output
                    drawGrid(leftR, fine, fine, false, false, colorForValue, textForValue);
                    drawGrid(rightR, 11, 11, true, true, colorForValue, textForValue);
                    break;
                case 1: // error
                    drawGrid(leftR, fine, fine, false, false, colorForError, textForError);
                    drawGrid(rightR, 11, 11, true, true, colorForError, textForError);
                    break;
                case 2: // test
                    drawGrid(leftR, fine, fine, false, false, colorForTest, textForTest);
                    drawGrid(rightR, 11, 11, true, true, colorForTest, textForTest);
                    break;
                default: // none
                    DrawRectangleRec(leftR, BLACK);
                    drawGrid(rightR, 11, 11, true, true, colorForValue, textForValue);
                    break;
            }
        }

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
