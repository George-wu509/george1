

|          |                                                                  |
| -------- | ---------------------------------------------------------------- |
| 整体系统架构:  | 1. 边缘设备层(Edge Device)<br>   2. 本地服务器层(Server)<br>   3. 客户端/远程访问层 |
| 数据流与存储架构 | 1. 数据采集流程<br>   2. 数据存储与管理                                       |
| 软件架构     | 1. 控制软件层<br>   2. 分析软件层<br>   3. 云端集成的潜力                         |
| 系统设计评估   |    1. 设计优势<br>   2. 设计局限                                         |

# Sartorius Incucyte产品系统设计分析

Sartorius的Incucyte产品是一个先进的实时细胞成像与分析系统,从系统设计角度分析,它采用了一种特殊的分布式架构设计。本报告将深入分析其系统架构,特别关注云端、服务器、边缘设备与Incucyte产品间的关系。

## 整体系统架构

Incucyte系统采用一种基于<mark style="background: #ABF7F7A6;">边缘计算</mark>和<mark style="background: #ABF7F7A6;">本地服务器</mark>的混合架构,主要由三部分组成:

## 边缘设备层(Edge Device)

Incucyte的显微镜部分作为边缘设备,直接在数据生成处(细胞培养箱内)进行操作:

- **显微镜单元**: 配备高分辨率摄像头(如Basler Ace 1920-155um CMOS),可直接置于标准细胞培养箱内[2](https://www.fredhutch.org/content/dam/www/shared-resources/ci/IncuCyte_May2023.pdf)
    
- **光学系统**: 包含自动转盘上的多个物镜(4X、10X、20X),支持相差显微和荧光成像[2](https://www.fredhutch.org/content/dam/www/shared-resources/ci/IncuCyte_May2023.pdf)[11](https://www.sartorius.com/en/products/live-cell-imaging-analysis/live-cell-analysis-instruments/s3-live-cell-analysis-instrument)
    
- **样品管理**: 能同时容纳多个实验容器,最多支持六个微孔板并行处理[2](https://www.fredhutch.org/content/dam/www/shared-resources/ci/IncuCyte_May2023.pdf)
    
- **数据采集**: 在培养箱稳定环境中直接采集细胞图像,减少外部干扰[1](https://www.sartorius.com/en/products/live-cell-imaging-analysis)
    

这一边缘设备设计确保了细胞样本在最小干扰下被持续监测,解决了传统细胞成像需要将样本从培养箱移出的问题,极大降低了环境变化对实验的影响。

## 本地服务器层(Server)

Incucyte Controller作为本地服务器,负责系统控制和数据处理:

- **硬件配置**: 配备高性能计算硬件,包括16.4 TB RAID硬盘阵列和48 GB RAM[15](https://www.sartoriustr.com/Upload/Dosyalar/resim-pdf/incucyte-s3-live-cell-analysis-be0e4771-2a57-45b8-834f-ab78943ece09.pdf)
    
- **操作系统**: 运行64位Windows 10[15](https://www.sartoriustr.com/Upload/Dosyalar/resim-pdf/incucyte-s3-live-cell-analysis-be0e4771-2a57-45b8-834f-ab78943ece09.pdf)
    
- **网络连接**: 支持10/100/1000 Mbps以太网连接[15](https://www.sartoriustr.com/Upload/Dosyalar/resim-pdf/incucyte-s3-live-cell-analysis-be0e4771-2a57-45b8-834f-ab78943ece09.pdf)
    
- **数据处理**: 执行图像处理、分析和存储功能
    
- **用户界面**: 提供触摸屏界面用于直接控制和监测[8](https://www.sartorius.com/en/products/live-cell-imaging-analysis/live-cell-analysis-servicing-support)
    

服务器层处理从边缘设备收集的大量图像数据,执行初步分析,并提供本地存储解决方案,确保数据处理的低延迟和高效率。

## 客户端/远程访问层

系统支持多用户远程访问,使研究人员能够从任何网络连接的计算机监控和控制实验:

- **远程连接机制**: 通过IP地址或主机名连接到Incucyte Controller[8](https://www.sartorius.com/en/products/live-cell-imaging-analysis/live-cell-analysis-servicing-support)
    
- **用户授权**: 支持多用户账户和权限管理,提供无限用户许可证[11](https://www.sartorius.com/en/products/live-cell-imaging-analysis/live-cell-analysis-instruments/s3-live-cell-analysis-instrument)[12](https://medicine.biu.ac.il/sites/medicine/files/shared/ZABAM/docs/IncuCyte%20User%20Manual%202020Bziped.pdf)
    
- **跨平台访问**: 主要支持Windows系统,Mac用户需通过特殊门户访问[6](https://www.uib.no/sites/w3.uib.no/files/attachments/incucyte_s3_user_manual_per19012024.pdf)
    
- **数据访问**: 允许远程查看实验进度、分析数据和调整参数
    

## 数据流与存储架构

## 数据采集流程

1. **自动采集**: 系统按预设时间间隔在培养箱内自动采集细胞图像
    
2. **多模态数据**: 同时获取相差显微图像和最多三种荧光通道的图像[17](https://micr.med.wayne.edu/incucyte)
    
3. **实时传输**: 数据从显微镜(边缘设备)实时传输至Controller(本地服务器)
    
4. **初步处理**: 在Controller上进行图像拼接、校准和初步分析
    

## 数据存储与管理

1. **本地存储**: 主要数据存储在Controller的RAID硬盘阵列上(16.4 TB)[15](https://www.sartoriustr.com/Upload/Dosyalar/resim-pdf/incucyte-s3-live-cell-analysis-be0e4771-2a57-45b8-834f-ab78943ece09.pdf)
    
2. **数据归档**: 系统提供Archive功能,允许将历史数据移至外部存储位置[5](https://www.sartorius.com/download/1087348/incucyte-live-cell-analysis-systems-user-manual-en-l-8000-04-1--data.pdf)[14](https://medicine.biu.ac.il/sites/medicine/files/shared/ZABAM/docs/How%20to%20Archive.pdf)
    
3. **数据格式**: 支持多种图像格式(JPEG、PNG、TIFF)和视频格式(WMV、AVI、MPEG-4)导出[15](https://www.sartoriustr.com/Upload/Dosyalar/resim-pdf/incucyte-s3-live-cell-analysis-be0e4771-2a57-45b8-834f-ab78943ece09.pdf)
    
4. **数据安全**: 实现用户权限控制和工作组管理,保护敏感数据[5](https://www.sartorius.com/download/1087348/incucyte-live-cell-analysis-systems-user-manual-en-l-8000-04-1--data.pdf)
    

## 软件架构

## 控制软件层

Incucyte控制软件采用多层架构:

- **用户界面层**: 提供引导式操作界面,简化实验设置[3](https://www.sartorius.com/en/products/live-cell-imaging-analysis/live-cell-analysis-software)
    
- **调度层**: 管理多用户、多实验的扫描调度[11](https://www.sartorius.com/en/products/live-cell-imaging-analysis/live-cell-analysis-instruments/s3-live-cell-analysis-instrument)
    
- **设备控制层**: 直接控制显微镜硬件的低级功能
    
- **数据采集层**: 负责图像获取和初步处理
    

## 分析软件层

- **图像处理引擎**: 执行自动化图像分析,包括细胞识别和跟踪
    
- **AI分析模块**: 提供AI驱动的分析能力,如细胞分类和形态学分析[3](https://www.sartorius.com/en/products/live-cell-imaging-analysis/live-cell-analysis-software)
    
- **可视化工具**: 生成各种图表和报告,支持实时数据可视化[3](https://www.sartorius.com/en/products/live-cell-imaging-analysis/live-cell-analysis-software)
    
- **数据导出**: 允许将分析结果导出为多种格式
    

## 与云端集成的潜力

虽然Incucyte主要基于本地服务器架构,但系统设计显示出与云平台集成的潜力:

- **远程访问基础**: 现有的远程访问机制为云集成提供基础[6](https://www.uib.no/sites/w3.uib.no/files/attachments/incucyte_s3_user_manual_per19012024.pdf)[8](https://www.sartorius.com/en/products/live-cell-imaging-analysis/live-cell-analysis-servicing-support)
    
- **数据归档接口**: 数据归档功能可扩展为云存储集成[5](https://www.sartorius.com/download/1087348/incucyte-live-cell-analysis-systems-user-manual-en-l-8000-04-1--data.pdf)[14](https://medicine.biu.ac.il/sites/medicine/files/shared/ZABAM/docs/How%20to%20Archive.pdf)
    
- **分布式分析**: 随着数据量增长,可将高级分析任务迁移至云平台[9](https://www.redpanda.com/blog/streaming-data-platform-for-iot-edge)
    

从IoT架构角度看,Incucyte系统可以扩展成更完整的三层架构:

1. **Tier 1(设备层)**: Incucyte显微镜作为数据采集设备
    
2. **Tier 2(数据摄取层)**: Controller作为数据汇聚和预处理节点
    
3. **Tier 3(处理层)**: 可扩展至云平台进行大规模数据存储和高级分析[9](https://www.redpanda.com/blog/streaming-data-platform-for-iot-edge)
    

## 系统设计评估

## 设计优势

- **环境稳定性**: 通过在培养箱内进行成像,最大限度减少对细胞的干扰[1](https://www.sartorius.com/en/products/live-cell-imaging-analysis)
    
- **长时间监测**: 支持数天、数周甚至数月的连续细胞监测[1](https://www.sartorius.com/en/products/live-cell-imaging-analysis)
    
- **多用户协作**: 无限用户许可和远程访问支持团队协作[11](https://www.sartorius.com/en/products/live-cell-imaging-analysis/live-cell-analysis-instruments/s3-live-cell-analysis-instrument)
    
- **可扩展应用**: 模块化设计支持多种应用场景,从基础细胞计数到复杂的免疫细胞分析13
    

## 设计局限

- **有限扩展性**: 存储和计算能力受本地硬件限制
    
- **网络依赖**: 远程访问依赖稳定的网络连接
    
- **云集成不完善**: 缺乏与主流云平台的原生集成
    
- **数据共享挑战**: 跨机构数据共享机制有限
    

## 结论

Sartorius的Incucyte产品从系统设计角度看,采用了一种优化的"边缘设备-本地服务器"架构,非常适合实验室环境中的细胞成像和分析需求。该设计平衡了实时数据采集、本地处理和远程访问的需求,为生物研究提供了强大的工具。

未来发展方向可能包括加强与云平台的集成,增强分布式分析能力,以及开发更强大的跨机构协作功能。这将使Incucyte系统从当前的实验室级解决方案扩展为更全面的生物研究数据平台。