# Curvature-based Analysis of Network Connectivity in Private Backbone Infrastructures

This repository provides the tool to replicate most of the studies from the paper: “Curvature-based Analysis of Network Connectivity in Private Backbone Infrastructures.”



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Installation
Installation

To get started with the repository, clone it to your local machine and install the necessary dependencies:

```sh
git clone https://github.com/yourusername/Curvature-based-Analysis-of-Network-Connectivity.git
cd Curvature-based-Analysis-of-Network-Connectivity
pip install -r requirements.txt
```


<!-- USAGE EXAMPLES -->
## Usage

After installing the dependencies, you can collect the latency measurements for all the anchor meshes for a given date by updating the Date util.py script and run the full_pipeline.py script:

```sh
python full_pipeline.py
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

Since Aug. 2023, both AWS and Google are not hosting RIPE Atlas anchors anymore. The code is still functional, but it is not possible to collect data for 2023 and beyond. If you are interested in running the technique for mapping the cloud, you can use the approach described in the paper for Azure (that is opening VMs in different regions and measuring the latency between them). That's more complex and we are currently working toward automating this process such that we can provide latency directly for utilization and avoid the massive overhead.


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

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
