#version 330 core
out vec4 FragColor;

in vec3 Normal;  
in vec3 FragPos;  
in vec2 TexCoord;
in float outColor;
  
uniform vec3 lightPos; 
uniform vec3 lightColor;
uniform vec3 objectColor;
uniform int applyTexture;

uniform sampler2D textureID;

void main()
{
    // ambient
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;
  	
    // diffuse 
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = abs(dot(norm, lightDir));
    vec3 diffuse = diff * lightColor;
            
    vec3 result = (ambient + diffuse) * objectColor;

    if(outColor > -2){
        FragColor = vec4(vec3(outColor, 0, 0), 1.0);
    }else if(applyTexture == 1){
        FragColor = vec4(texture(textureID, TexCoord).rgb * result, 1.0);
    }else{
        FragColor = vec4(result, 1.0);
    }
} 

