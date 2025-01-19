
> [!NOTE] 
> My idea of what an implementation of WeatherMesh would look like. Based on https://windbornesystems.com/blog/weathermesh-2-technical-blog. 


# Notes on WeatherMesh
- Dataset: ERA-5 + Others
- Input: weather variables on a regular grid around the entire globe
- Output: the entire weather state at a future time
  - Geopotential, temperature, winds, and moisture at 25 vertical levels
  - More variables at surface level
  - 

- Model architecture is a standard **encoder-processor-decoder** architecture
- Interesting thing: keeping the state of the atmosphere in the latent space
  -  Avoids accumulating errors when converting to/from latent space
-  