

|                                        |     |
| -------------------------------------- | --- |
| [[Video upload and analysis platform]] |     |

![[Pasted image 20250808152526.png]]



|                                        |                                                                               |     |
| -------------------------------------- | ----------------------------------------------------------------------------- | --- |
| DATA                                   | AWS S3                                                                        |     |
|                                        |                                                                               |     |
| ML pipeline 1<br>(Video preprocessing) | AWS Step functions<br>AWS Lambda<br>AWS Fargate<br>AWS Elamental MediaConvert |     |
| ML pipeline 2<br>(Parallel)            | AWS Rekognition Video<br>AWS SageMaker<br>AWS SageMaker Neo                   |     |
| ML pipeline 3<br>(Store)<br>           | AWS DynamoDB                                                                  |     |
|                                        |                                                                               |     |
| MLOPs 1<br>(Experiment tracking)       | Wandb<br>Clearml<br><br>AWS SageMaker Experiments                             |     |
| MLOPs 2<br>(Model version control)     | AWS SageMaker Model registry                                                  |     |
| MLOPs 3<br>(CI/CD)                     | Github Action<br><br>AWS SageMaker Pipeline                                   |     |
| MLOPs 4<br>(MLOPs)                     | AWS SageMaker Model Monitor<br>Amazon EventBridge                             |     |
