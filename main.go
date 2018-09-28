package main

import (
	"fmt"
	"log"
	"path/filepath"

	"github.com/Kagami/go-face"
)

const dataDir = "images"

func main() {
	fmt.Println("Facial Recognition System v0.01")

	rec, err := face.NewRecognizer(dataDir)
	if err != nil {
		fmt.Println("Cannot initialize recognizer")
	}
	defer rec.Close()

	fmt.Println("Recognizer Initialized")

	avengersImage := filepath.Join(dataDir, "avengers-02.jpeg")

	faces, err := rec.RecognizeFile(avengersImage)
	if err != nil {
		log.Fatalf("Can't recognize: %v", err)
	}
	fmt.Println("Number of Faces in Image: ", len(faces))

	var samples []face.Descriptor
	var avengers []int32
	for i, f := range faces {
		samples = append(samples, f.Descriptor)
		// Each face is unique on that image so goes to its own category.
		avengers = append(avengers, int32(i))
	}
	// Name the categories, i.e. people on the image.
	labels := []string{
		"Dr Strange",
		"Tony Stark",
		"Bruce Banner",
		"Wong",
	}
	// Pass samples to the recognizer.
	rec.SetSamples(samples, avengers)

	// Now let's try to classify some not yet known image.
	testTonyStark := filepath.Join(dataDir, "tony-stark.jpg")
	tonyStark, err := rec.RecognizeSingleFile(testTonyStark)
	if err != nil {
		log.Fatalf("Can't recognize: %v", err)
	}
	if tonyStark == nil {
		log.Fatalf("Not a single face on the image")
	}
	avengerID := rec.Classify(tonyStark.Descriptor)
	if avengerID < 0 {
		log.Fatalf("Can't classify")
	}

	fmt.Println(avengerID)
	fmt.Println(labels[avengerID])

	testDrStrange := filepath.Join(dataDir, "dr-strange.jpg")
	drStrange, err := rec.RecognizeSingleFile(testDrStrange)
	if err != nil {
		log.Fatalf("Can't recognize: %v", err)
	}
	if drStrange == nil {
		log.Fatalf("Not a single face on the image")
	}
	avengerID = rec.Classify(drStrange.Descriptor)
	if avengerID < 0 {
		log.Fatalf("Can't classify")
	}

	fmt.Println(avengerID)
	fmt.Println(labels[avengerID])

	testWong := filepath.Join(dataDir, "wong.jpg")
	wong, err := rec.RecognizeSingleFile(testWong)
	if err != nil {
		log.Fatalf("Can't recognize: %v", err)
	}
	if wong == nil {
		log.Fatalf("Not a single face on the image")
	}
	avengerID = rec.Classify(wong.Descriptor)
	if avengerID < 0 {
		log.Fatalf("Can't classify")
	}
	fmt.Println(avengerID)
	fmt.Println(labels[avengerID])

}
