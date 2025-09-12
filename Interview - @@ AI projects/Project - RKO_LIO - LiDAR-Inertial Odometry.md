
linkedin post [link](https://www.linkedin.com/posts/cyrill-stachniss-736233173_robotics-lidar-odometry-activity-7371911002491322371-WYXo?utm_source=share&utm_medium=member_desktop&rcm=ACoAABMNj2MBAJSP3cWd4xpiz4wB7qdx43hvW18)

![[Pasted image 20250912003102.png]]

Interested in LiDAR-inertial odometry without needing to dive into sensor-specific modeling? Then, we have released a new work by [Meher V. Ramakrishna Malladi](https://www.linkedin.com/in/mvrmalladi/), ready to be used. Python and ROS versions are available, covering all currently supported ROS distributions.  
  
The work is described in detail in "A Robust Approach for LiDAR-Inertial Odometry Without Sensor-Specific Modeling" by [Meher V. Ramakrishna Malladi](https://www.linkedin.com/in/mvrmalladi/), [Tiziano Guadagnino](https://www.linkedin.com/in/tiziano-guadagnino-087119170/), [Luca Lobefaro](https://www.linkedin.com/in/luca-lobefaro-1ab872a9/), and [Cyrill Stachniss](https://www.linkedin.com/in/cyrill-stachniss-736233173/), 2025  
  
Video: [https://lnkd.in/gBV5iw4J](https://lnkd.in/gBV5iw4J)  
Paper: [https://lnkd.in/g355h3GU](https://lnkd.in/g355h3GU)  
Code: [https://lnkd.in/gk9uDzU9](https://lnkd.in/gk9uDzU9)  
The github repository has an extensive README.  
  
The Python version can be installed with a simple:  
pip install rko_lio  
  
and used with  
rko_lio --viz /path/to/rosbag  
  
The Python version builds on Ubuntu 22/24, macOS 14/15, and Windows; it builds on all OS for both x64 and ARM architectures. For ROS, we support the Distros: Humble, Jazzy, Kilted, and Rolling.