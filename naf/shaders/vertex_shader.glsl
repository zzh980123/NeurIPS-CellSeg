#version 330

in vec3 in_position;
in vec2 in_texcoord_0;
out vec2 vTexCoord;

void main(){
    gl_Position = vec4(in_position, 1);
    vTexCoord = in_texcoord_0;
}