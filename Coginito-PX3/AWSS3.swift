import SwiftUI
import AWSS3

func uploadData(data: Data, name:String)->String{
    var results_ = ""
    
    let transferUtility = AWSS3TransferUtility.default()
    // アップロードするバケット名/アップしたいディレクトリ
    let bucket = "pacifista-px3"
    // ファイル名
    let key = name + ".png"
    let contentType = "application/png"
    // アップロード中の処理
    let expression = AWSS3TransferUtilityUploadExpression()
    expression.progressBlock = {(task, progress) in
       DispatchQueue.main.async {
         // アップロード中の処理をここに書く
       }
    }
    
    // アップロード後の処理
    let completionHandler: AWSS3TransferUtilityUploadCompletionHandlerBlock?
    completionHandler = { (task, error) -> Void in
       DispatchQueue.main.async {
         if let error = error {
             fatalError(error.localizedDescription) // 失敗
         } else {
            // アップロード後の処理をここに書く
         }
       }
     }
        
     // アップロード
     transferUtility.uploadData(
       data,
       bucket: bucket,
       key: key,
       contentType: contentType,
       expression: expression,
       completionHandler: completionHandler
     ).continueWith { (task) -> Any? in
       if let error = task.error as NSError? {
           results_ = "upload failed"
           fatalError(error.localizedDescription)
       } else {
           // アップロードが始まった時の処理をここに書く
           results_ = "upload succeed"
       }
       return nil
     }
    return results_
}
