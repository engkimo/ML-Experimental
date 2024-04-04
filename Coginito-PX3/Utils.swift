import SwiftUI
import Vision

func calcurateTime(stime:Date)->String{
    let timeInterval = Date().timeIntervalSince(stime)
    return String((timeInterval * 100) / 100) + "[ms]"
}

func trimmingImage(_ image: UIImage, ratio_:Double)-> UIImage{
    let trimmingArea = CGRect(x:0.0, y:0.0, width:image.size.width*ratio_, height:image.size.height*ratio_)
    let imgRef = image.cgImage?.cropping(to: trimmingArea)
    let trimImage = UIImage(cgImage: imgRef!, scale: image.scale, orientation: image.imageOrientation)
    return trimImage
}
