<h1 align="center" style="font-weight: 500; line-height: 1.4;">
  <strong>AMOD</strong>: <strong>A</strong>RMA3 <strong>M</strong>ilitary <strong>O</strong>bject <strong>D</strong>etection
</h1>

<p align="center">
  <a href="#"><img alt="ARMA3" src="https://img.shields.io/badge/Game-ARMA3-red?logo=steam"></a>
  <a href="./blob/main/LICENSE"><img alt="MIT" src="https://img.shields.io/badge/License-MIT-green?logo=MIT"></a>
</p>

<p align="center">
  <b><a href="github.com/unique-chan">Yechan Kim</a></b> and
  <b><a href="github.com/citizen135">JongHyun Park</a></b>
</p>


### This repo includes:
- Introduction to AMOD dataset series


### Preview
<p align="center">
    <img alt="Welcome" src="./figs/sample.png" />
</p>

### Updates
- (09/2023) TBD


### Open dataset
- You can easily access to our open dataset for quick research:
  - [![Download AMOD-v1](https://img.shields.io/badge/Download_(Available_later!)-AMOD--v1-gray?color=red)](#)

### Dataset structure
- The directory structure of our dataset is as follows:
~~~
|—— 📁 {train or test}_{map_name}_{weather}_{start_hour}_{end_hour}_...
	|—— 📁 0000 (scene number)
		|—— 📁 -20  (look angle)
			|—— 🖼️ EO_0000_-20.png  
			|—— 📄 ANNOTATION_0000_-20.csv (including bbox labels)
		|—— 📁 +20 
			|—— 🖼️ ...
	|—— 📁 0001
		|—— 📁 -20
		|—— 📁 ... 
	|—— 📁 0002
		|—— 📁 -20
		|—— 📁 ...
	...
	|—— 📄 meta_..._.csv (including in-game shooting time, weather, and error logs per each scene)
~~~
- You may need to transform the above folder structure before training your own model.
- You can conveniently check the data using our [image viewer](https://github.com/Dodant/AMOD-viewer).



### Citation
- Paper is coming soon!


### Contribution
- If you find any bugs for further improvements, please feel free to create issues on GitHub!
- All contributions and suggestions are welcome. Of course, stars (🌟) are always welcome.
