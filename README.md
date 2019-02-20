
<img src="https://github.com/zstern/CNN_PCI/blob/master/presentation/texas_pothole.jpg">





_Pothole the size of Texas. Credit: Daniel Lobo (https://www.flickr.com/photos/daquellamanera/3038436394)_

**<span style="text-decoration:underline;">Building a convolutional neural network to categorize and evaluate road quality.  </span>**

**<span style="text-decoration:underline;">Problem Definition: </span>**

The overall quality of roads in the United States is poor and getting poorer.  The American Society of Civil Engineers (ASCE) gives the roads in the United States a grade of 'D' for quality.  This poor road quality costs drivers money by causing poor gas mileage and increased maintenance costs.  It is estimated that the poor road quality in Washington, DC costs drivers there an additional $515 per year on average in additional maintenance.   

This project has two purposes:  While the economic impact of road quality has been well documented, it is harder to find information on the effects of road quality on public safety.  I wanted to see if I could find any correlation between road quality and traffic accidents that resulted in injuries.  Second, I wanted to see if I could build a tool which could aid in assessing road quality by using a convolutional neural networks and available photo data to grade roads based on an established road grading index.  Perhaps, if road quality assessments could be made faster and easier, road maintenance could be improved.

There are several measures of road quality but the one that this project focuses on is known as the "Pavement Condition Index" or just PCI.  The PCI:



*   Provides a numerical rating for the condition of road segments within the road network, where 0 is the worst possible condition and 100 is the best. 
*   The PCI measures two conditions: 
    *   The type, extent and severity of pavement surface distresses (typically cracks and rutting)
*   The PCI is a subjective method of evaluation based on inspection and observation. 
*   A PCI rating is developed by having experienced public works officials drive the road network and evaluate its condition in a systematic way. 

Since the PCI is essentially a visual inspection of the road, that does not involve subsurface investigation or specialized machinery,  I theorized that it would be possible to automate the PCI  inspection process using computer vision.  

**<span style="text-decoration:underline;">Gathering the Data:</span>**

Washington,DC has a data portal for publicly available datasets.  Specifically I was interested in traffic accident data which is available here: [http://opendata.dc.gov/datasets/crashes-in-dc](http://opendata.dc.gov/datasets/crashes-in-dc)  

The purpose of using this dataset was that it provided a listing of geographical points within DC that I could use as a basis for gathering further data about those points (the road quality and photos of the roads at those points). It also allows us to create a data set where we can combine accident data with pavement condition data to evaluate accidents in relation to pavement conditions.  

The "crashes in DC" dataset has approximately 117,000 lines of data, where each line represents a traffic accident (but not necessarily an accident that caused injury).  A few points of data go back to 1975 but is mostly focused between 2013 to the present.  I have focused on the 2013 to the present time period.  

The crashes data set provides us with the geographic location of the accident both using latitude and longitude and also an X/Y coordinate using a local coordinate system ([http://epsg.io/26985](http://epsg.io/26985)) .  The X/Y coordinate system is important because it is how the DC Department of Transportation has made available the road quality data.  Since the road quality data is not part of the crashes data set I needed to gather and collate the road quality data, by location and join them to the crash dataset.  DC has a website where you can query a location and be provided with data on the type of road, the centerline info and ultimately the road quality data including PCI, which is what I was most interested in. The PCI is available as a number on a 0 to 100 scale and also binned into poor,fair,good or excellent.  

([https://rh.dcgis.dc.gov/dcgis/rest/services/DDOT/RoadwayBlocks/MapServer/exts/DdotLrsExtensions/getPointOnRoute](https://rh.dcgis.dc.gov/dcgis/rest/services/DDOT/RoadwayBlocks/MapServer/exts/DdotLrsExtensions/getPointOnRoute))

I used the crash data set to ping the api at each crash location to get the road quality data, point by point and then merged the data together into one final data set of approximately 27,000 points (after further cleaning).  

The second part of the project involved gathering photos of the roads at the specific locations.  For this section of the project I only used approximately 10,000 data points, selected at random from the larger set and with the four pci classes balanced.  I then queried the google street view api for each point of interest.  The street view api allows for the user to specify an orientation of the photo (how far the image is pointed up or down), so during the api call, I specified a downward angle to gather a better view of the street.  The photos were downloaded at 300 x 300 pixels.  

Two examples of the street view photos:

(Google street view photo corresponding to a PCI of "poor")

<p id="gdcalert2" > <img src="https://github.com/zstern/CNN_PCI/blob/master/presentation/24443725_poor.jpg"
    </p>





(Google street view photo corresponding to a PCI of "excellent")

<p id="gdcalert2" > <img src="https://github.com/zstern/CNN_PCI/blob/master/presentation/24128720_ex.jpg"
    </p>




**<span style="text-decoration:underline;">Exploring and Modeling the data:</span>**

For the first portion of the project I wanted to look at the interaction of pci and accidents that resulted in injuries.  The original data set lists the number of people injured by type of person injured (driver injury, pedestrian injury, bicyclist injury) and the severity (minor, major or fatal).  Since I was just interested in whether an injury happened or not, I created a new column that only tracked if any injury to any person had happened in the accident, and if so, and record it as a 1 or 0 (1 means an injury too place, 0 means there were no injuries).  

My hypothesis test statements were:

H0: Road quality does NOT have an affect on the incidence of injuries during accidents

H1: Road quality does have an affect on the incidence of injuries during accidents. 

To test the hypothesis I used the Statsmodels package to run a logistic regression using the pci index (0-100) to predict whether injuries took place or not (1 or 0).  The model showed that the p-value of the pci coefficient was significant at the 1% level (p-value of 0.001). Therefore we reject the null hypothesis and accept the alternative hypothesis.  We conclude that road quality does have an affect on the incidence of injuries at a statistically significant level.  Though upon further examination of the model, things are not as simple as they seem.  

The coefficient for the pci variable is positive, meaning that a positive increase in pci quality on the 0-100 scale corresponds to a positive increase in injury accidents.  This is counterintuitive, in that we would expect an increase in quality to result in a decrease in injuries.  My theory here is that there are confounding factors that go along with high road quality.  Perhaps roads with higher quality are also main roads, or roads where people generally drive faster, or they get much more traffic (and are maintained more regularly).  Those factors, such as more people driving and at a higher rate of speed (but on better roads) result in more injuries when accidents do occur.  

For the second portion of the project I wanted to see if I could build a model that would correctly categorize road quality based on only a photo of the road. To do this I took the Google street view photos and created labeled data sets based on the corresponding pci rating for each point.   

The PCI rating of the roads are binned into four categories: poor, fair, good, excellent.  My first model attempt was to try to correctly categorize and distinguish between the best and worst rated roads (poor vs excellent).   

To do this I created a convolutional neural network using the keras library.  The input data to this network were the photos from each class; approximately 2,700 photos of each class.  The data was run through the CNN for 100 epochs.  The result was an accuracy of 50%, meaning that the model had zero predictive ability in discerning pavement quality between the poor and excellent classes.  

I hypothesized that the photos did not have enough contrast in them and perhaps the model would be better if I preprocessed the images to increase the contrast.  If I increased the contrast then maybe the road imperfections would become more visible.  

One way of interpreting contrast is to think of the histogram of pixel values in a photo.  In a low contrast photo the pixel values are all bunched up in a smaller range.  Increasing contrast spreads the pixels out over a greater range of values (pushing pixels to become more black or white instead of a grey middle value).  

Increase contrast in a photo can be done 'globally', that is, over the entire photo, or can be done locally (on smaller sections of the photo).  If you increase contrast globally you may end up pushing some pixels all the way to white or black and thus you would lose all detail in that section.  To combat this I used an algorithm called CLAHE which stands for Contrast Limited Adaptive Histogram Equalization.  CLAHE takes a smaller subset of the photo and calculates how to spread out the pixels for that one region.  It then goes to each subsection of the photo and repeats until the entire photo has been processed.  CLAHE lets us increase the contrast in the photo but also preserves details that may have been lost had we just increased the contrast as a whole.  Below are the same two pictures from before, after having been processed with CLAHE:



<p id="gdcalert2" > <img src="https://github.com/zstern/CNN_PCI/blob/master/presentation/C_24443725_p.jpg"
    </p>



  



<p id="gdcalert2" > <img src="https://github.com/zstern/CNN_PCI/blob/master/presentation/C_24128720_e.jpg"
    </p>


In the above examples it appears that the damage and cracks on the poor quality road were now much more pronounced.  

I processed all the photos using CLAHE and re-ran them through the CNN for 100 epochs.  I also added an additional filter and pooling layer within the network.  Again the results were the same:  50% accuracy in categorizing the photos between two classes i.e the model provided no functional value.  

<span style="text-decoration:underline;">Issues with the data and potential improvements:</span>

Google street view photos, while convenient for the purposes of this project, provide dirty and noisy data.  There are watermarks, overlays, ghosting images and cars halfway in the picture such as the photo below: 



<p id="gdcalert2" > <img src="https://github.com/zstern/CNN_PCI/blob/master/presentation/24290072.jpg"
    </p>


The photos that I choose to show above were picked precisely because they are the best examples of what was in each category.  But this photo is much more representative of the type of photos that are in the sets overall.  Also, when gathering photos through an api, you may not always get a full picture of the road and may get an angle that only shows a building or sidewalk. 

Evaluating a road using PCI is typically done for a whole road section (ex. Taking a 100 meter stretch of road to evaluate in its entirety). It is not done on a snapshot basis.  If I had purposely gathered data, perhaps using high quality videos of stretches of road, the CNN would more likely have been able to distinguish what the road defects were.  There are a few categories of road defect that people are to look for when evaluating a road on the PCI scale, such as "loss of aggregate", "distortion" and "cracking".  If we were to create a labeled training set purposely to evaluate larger road conditions using the PCI evaluation method, I believe that we could create a model that provides a usable amount of accuracy.   A system could operate using an off the shelf HD video camera that faces the road.  For each road defect found through the camera the system would count those defects accordingly.  The benefits of such a system would be:



*   Reduced cost from decreased manpower needed. PCI practices state that two people should be involved; one to drive and one to survey the road.  A CV system would only require a driver.  
*   Increased standardization of ratings:  PCI is subject to the proclivities of the evaluator as it is a subjective, qualitative assessment of road condition. Using computer vision, we could create a more quantitative standard.  
*   Increased speed: And because of increased speed, evaluations could be done more often. 

While my model failed to provide any predictive value, I believe that it is entirely feasible to gather data in a more methodical and purposeful way, and in doing so, allow a model to better categorize the road quality correctly.  

<span style="text-decoration:underline;">References</span>:

[http://hawaiiasphalt.org/wp/wp-content/uploads/PCI-101.pdf](http://hawaiiasphalt.org/wp/wp-content/uploads/PCI-101.pdf)

[https://www.infrastructurereportcard.org/cat-item/roads/](https://www.infrastructurereportcard.org/cat-item/roads/)

[https://www.wired.com/2015/07/heres-much-citys-crappy-roads-costing-you%E2%80%A8/](https://www.wired.com/2015/07/heres-much-citys-crappy-roads-costing-you%E2%80%A8/)

[https://www.washingtonpost.com/news/wonk/wp/2015/06/25/why-driving-on-americas-roads-can-be-more-expensive-than-you-think/?utm_term=.5f92204dac62](https://www.washingtonpost.com/news/wonk/wp/2015/06/25/why-driving-on-americas-roads-can-be-more-expensive-than-you-think/?utm_term=.5f92204dac62)

[https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html)


<!-- Docs to Markdown version 1.0β14 -->
