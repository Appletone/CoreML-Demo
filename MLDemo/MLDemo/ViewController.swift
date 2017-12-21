//
//  ViewController.swift
//  MLDemo
//
//  Created by louis on 2017/9/30.
//  Copyright © 2017年 louis. All rights reserved.
//

import UIKit
import CoreML

class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        let model = DTs()
        guard let input_data = try? MLMultiArray(shape:[1], dataType:.double) else {
            fatalError("Unexpected runtime error. MLMultiArray")
        }
        input_data[0] = 0
        input_data[1] = 0
        
        
        if let pred = try? model.prediction(input: input_data) {
            print(pred.classLabel)
            print(pred.classProbability)
        }
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }


}

