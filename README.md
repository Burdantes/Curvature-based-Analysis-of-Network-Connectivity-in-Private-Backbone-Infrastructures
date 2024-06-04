# Curvature-based Analysis of Network Connectivity in Private Backbone Infrastructures

This repository provides the tool to replicate most of the studies from the paper: “Curvature-based Analysis of Network Connectivity in Private Backbone Infrastructures.”



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Installation
Installation

To get started with the repository, clone it to your local machine and install the necessary dependencies. You can do this by running the following commands in your terminal. The code has been tested on Python 3.9:

```sh
git clone https://github.com/yourusername/Curvature-based-Analysis-of-Network-Connectivity.git
cd Curvature-based-Analysis-of-Network-Connectivity
pip install -r requirements.txt
```


<!-- USAGE EXAMPLES -->
## Usage

After installing the dependencies, you can collect the latency measurements for all the anchor meshes for a given date by updating the **start_date** in the util.py script (it is automatically set at **start_date = '2023-01-01'**). You can also mention which cloud you would like to focus on (AWS or Google) by updating the variable **which_cloud** to *aws* or *google*, you can then run the full_pipeline.py script: 

```sh
python full_pipeline.py
```
This process collects the anchor measurements across a week (that is a pretty slow process, feel free to crank up the number of processes) and saves the smallest latency measurements for each anchor pair in the **Datasets/AnchorMeasurements/$start_date** folder. Then it automatically generates the residual latency and associate graph. It also computes the graphs that it stores in **Datasets/Graphs/$start_date**. These graphs can be then analyzed using the scripts in the **Analysis** folder.

```sh
python boxplot.py
python heatmap.py
python sankey.py
```

<!-- ROADMAP -->
## Roadmap

- [x] Clean out the core code
- [ ] Add the analysis script to generate Sankey diagrams
- [ ] Add the analysis script to generate the graph visuals
- [ ] Automate the manifold code.

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Update in 2023 - no more RIPE Atlas Anchors

Since August 2023, both AWS and Google no longer host RIPE Atlas anchors. While the code remains functional, it is no longer possible to collect data for 2023 and beyond using these platforms. If you are interested in using the technique for mapping the cloud, you can apply the approach described in the paper for Azure. This involves opening VMs in different regions and measuring the latency between them. This process is more complex, and we are currently working on automating it to provide latency data directly, thereby avoiding the significant overhead.

<!-- CONTRIBUTING -->
## Contributing

We welcome contributions to enhance the analysis and visualizations. To contribute:

1.	Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3.	Make your changes and commit them (git commit -am 'Add new feature'). 
4. Push to the branch (git push origin feature-branch). 
5. Create a new Pull Request.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - Loqman Salamatian (ls3748 at columbia dot edu)

Project Link: [https://github.com/burdantes/](https://github.com/Burdantes/Curvature-based-Analysis-of-Network-Connectivity-in-Private-Backbone-Infrastructures)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
