# PX3 - CoreML S3Uploader for ML dataset

# Overall : For collecting ML Dataset by iPhone or iPad (iOS)

This app is to collect ML Dataset by iPhone easily and save S3 bucket from remote area.

This system's overall is as follows:

<img src="https://user-images.githubusercontent.com/48679574/201074649-5a035b77-fcbb-4db7-a5c8-66eacd811b21.png" width="500" height="500"/>


# How to Use
1. install AWS API by podfile
2. make Amazon Cognito ID Pool at AWS Cognito and create bucket at S3 
3. select Photo
4. Predict image by ResNet50 as you like
5. input image name and label as upload name
6. upload photo and label with name to S3 bucket

# PX3 - Putting into action

<img src="https://user-images.githubusercontent.com/48679574/201076523-1a5bec58-a71b-4b23-93bb-05aa697b968b.gif" width="400" height="600"/>


# References
・[ Xcode/Swift - CocoaPodsの使い方を徹底解説](https://ios-docs.dev/cocoapods/)

・[SwiftからS3に画像をアップロードする方法](https://loooooovingyuki.medium.com/swiftからs3に画像をアップロードする方法-1388a6a5e251)
