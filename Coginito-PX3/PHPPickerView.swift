import SwiftUI
import PhotosUI

struct PHPickerView: UIViewControllerRepresentable{
    @Binding var isShowSheet: Bool
    @Binding var captureImage: UIImage?
    
    class Coordinator: NSObject, PHPickerViewControllerDelegate{
        var parent: PHPickerView
        
        init(parent: PHPickerView){
            self.parent = parent
        }
        // when hoto library is selected, run
        // for delegate
        func picker(
            _ picker: PHPickerViewController,
            didFinishPicking results: [PHPickerResult]){
            if let result = results.first{
                // get UIImage type photo
                result.itemProvider.loadObject(ofClass: UIImage.self){
                    (image, error) in
                    if let unwrapImage = image as? UIImage{
                        // add selected image
                        self.parent.captureImage = unwrapImage
                    } else {
                        print("No image avilable")
                    }
                }
            } else {
                print("No selected image")
            }
            // close sheet
            parent.isShowSheet = false
        }  // last picker block
    } // last Coordinator block
    // create Coordinator for import by SwifUI
    func makeCoordinator() -> Coordinator{
        // Coordinator instance
        Coordinator(parent:self)
    }
    // Run when View is cereated
    func makeUIViewController(context: UIViewControllerRepresentableContext<PHPickerView>)-> PHPickerViewController  {
        var configuration = PHPickerConfiguration()
        // not move image selecterd
        configuration.filter = .images
        configuration.selectionLimit = 1
        // create instance of "PHPickerViewController"
        let picker = PHPickerViewController(configuration: configuration)
        // create delegate
        picker.delegate = context.coordinator
        return picker
    }
    // when View is update, run
    func updateUIViewController(_ uiViewController: PHPickerViewController, context: UIViewControllerRepresentableContext<PHPickerView>)
    {
        // no process
    }
}// PHPickerView laast block
