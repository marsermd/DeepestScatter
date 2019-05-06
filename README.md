# DeepestScatter
Real-time photo-realistic clouds path tracer with Lorenz-Mie scattering.

This is an enchancement on paper by Disney Research [Deep Scattering](http://drz.disneyresearch.com/~jnovak/publications/DeepScattering/DeepScattering.pdf).

# State of the project
*Preparation*
- [x] Generate a sample cloud in Houdini
- [x] Exporter of volumetric data from Houdini
- [x] Importer of volumetric data for python and c++

*Ground Truth*
- [x] Naive implementation of path tracer that only takes into account [Beer-Lambert law](https://en.wikipedia.org/wiki/Beer%E2%80%93Lambert_law) and direct scattering 
  <img src="/images/naive_cloud_cube_rendering.png" width="200"/>
- [x] Interactive visualisation of results
- [x] Naive implementation that also takes into account [Lorenz-Mie](https://en.wikipedia.org/wiki/Mie_scattering) phase function.
- [x] Implement multiple scattering using Monte-Carlo method
- [x] Implement progressive path tracing   
- [x] Implement custom progressive path
- [x] Implement Reinhard tone mapping
- [x] Implement next event estimation   
  <img src="/images/multiple_scattering_1.png" width="500"/>   


*Dataset*
- [x] Generate more clouds
- [x] Gather dataset of (cloud_id, point_in_cloud, sun_direction, view_direction, radiance)
- [x] Gather dataset of samples around the point according to the original paper
- [x] Separate the dataset for training, test, validation with total size of 15\`000\`000; leave a few clouds out of those sets for visual evaluation

*Training*
- [x] Build a model in pytorch as described in the original paper
- [x] Train the model
- [x] Implement an efficient neural-based renderer in c++
- [ ] Compare to the original results   
  <p float="left">
  <img src="/images/neural_rendering_7000_meters.png" width="45%" />
  <img src="/images/neural_rendering_12000_meters.png" width="45%" /> 
</p>

*Research*
Experiments with different architectures.
- [ ] Try out different grids layout and evaluate convergence
- [ ] Light prebacking
- [ ] Progressive NN evaluation for light rebacking
- [ ] Evaluate radiocity only for the outer layers of the cloud
