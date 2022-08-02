#version 330

//layout(location = 1) out vec3 DistanceMap;
//layout (location=0) in vec3 position;
in vec2 vTexCoord;

uniform sampler2D texture0;
uniform float Beta;
uniform vec2 Offset;

out vec4 fragColor;

void main()
{
    vec3 this_pixel = texture(texture0, vTexCoord).rgb;
    vec3 east_pixel = texture(texture0, vTexCoord + Offset).rgb;
    vec3 west_pixel = texture(texture0, vTexCoord - Offset).rgb;

    // Squared distance is stored in the BLUE channel.
    float A = this_pixel.b;
    float e = Beta + east_pixel.b;
    float w = Beta + west_pixel.b;
    float B = min(min(A, e), w);

    // If there is no change, discard the pixel.
    // Convergence can be detected using GL_ANY_SAMPLES_PASSED.
    if (A == B) {
        discard;
    }

    fragColor.rg = west_pixel.rg;
    fragColor.b = B;

    // Closest point coordinate is stored in the RED-GREEN channels.
    if (A <= e && e <= w) fragColor.rg = this_pixel.rg;
    if (A <= w && w <= e) fragColor.rg = this_pixel.rg;
    if (e <= A && A <= w) fragColor.rg = east_pixel.rg;
    if (e <= w && w <= A) fragColor.rg = east_pixel.rg;
    fragColor = vec4(this_pixel, 1);
    //fragColor=vec4(1);
}