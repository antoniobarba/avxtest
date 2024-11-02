
float map_to(float value, float sourceMin, float sourceMax, float destMin, float destMax)
{
    float s = sourceMax - sourceMin;
    float d = destMax - destMin;
    float sourceRatio = (value - sourceMin) / s;
    return destMin + sourceRatio * d;
}

uint map_rgba(uchar r, uchar g, uchar b, uchar a)
{
    // return 0xffffffff;
    return r << 24 | g << 16 | b << 8 | a;
}

__kernel void calc_pixel(__global void *buffer, int w, int h)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    float x0 = map_to(i, 0, w, -2.0, 0.47);
    float y0 = map_to(j, 0, h, -1.12, 1.12);
    float x = 0;
    float y = 0;
    int iteration = 0;
    const int max_iteration = 1000;
    while (x * x + y * y <= 4 && iteration < max_iteration)
    {
        float xtemp = x * x - y * y + x0;
        y = 2 * x * y + y0;
        x = xtemp;
        ++iteration;
    }
    float color = map_to(iteration, 0, 15, 0, 255);
    __global uint * p = (__global uint *)buffer; 
    p[j * w + i] = map_rgba((uchar)color, (uchar)color, (uchar)color, 255);
}