Face2Text data readme

-----
Version
-----
1.0

-----
Release date
-----
22/September/2019

-----
Contents
-----
readme.txt: Information about this dataset.
raw.json: Dataset with all descriptions included and left unaltered as written by annotators.
(clean version to be included in future release)

-----
Paper
-----
Face2Text: Collecting an Annotated Image Description Corpus for the Generation of Rich Face Descriptions (http://www.lrec-conf.org/proceedings/lrec2018/summaries/226.html)

-----
Facebook
-----
https://www.facebook.com/Research-in-Vision-and-Language-group-RiVaL-114101215851716/

-----
Website
-----
http://rival.research.um.edu.mt/

-----
Stats
-----
Number of images: 4076
Number of descriptions: 5685
Maximum number of descriptions per image: 2
Minimum number ofdescriptions per image: 1
Maximum number of tokens in descriptions: 85
Minimum number of tokens in descriptions: 6
Characters used in descriptions: # '(),-./023578:;ABCDEFGHIJKLMNOPRSTWY]abcdefghijklmnopqrstuvwxyzéħż’

-----
Description
-----
Face2Text is an ongoing project to collect a dataset of natural language descriptions of human faces. 5000 randomly selected images were used from the CelebA dataset (https://www.kaggle.com/jessicali9530/celeba-dataset) and were presented to 4 different human annotators who each had a random sample of 2500 faces to describe. Note that linked paper does not describe this dataset as it was developed after the paper was published.

The dataset is provided in json format structured as a list of objects consisiting of the following:
    response_id: A unique ID referring to a unique annotator/image pair.
    filename: The filename of the image in the CelebA dataset.
    user_id: A unique ID referring to the annotator that wrote the description.
    description: The description written.

Here is an example:
    [
        {
            "response_id": "4683",
            "filename": "000035.jpg",
            "user_id": "2",
            "description": "A woman with a chiselled jaw, prominent cheekbones, a long, narrow nose and thin eyebrows. She has long, messy, black hair and she is wearing makeup."
        },
        ...
    ]

The images filename refers to a face photo in the CelebA dataset (https://www.kaggle.com/jessicali9530/celeba-dataset#img_align_celeba.zip). We do not include images here; you will need to download them and select them separately.

Instructions given to annotators were as follows:
    @ Describe these faces as naturally as possible.
    @ Do not spend too much time thinking about what to write. Just write the description which, in your view, accurately captures the physical attributes of the face.
    @ Don't describe the background and don't make inferences about the situation of the photo or the person (such as the person's job or background).
    @ You can describe a person's facial expression or their emotional state if this is evident from the picture.

-----
License
-----
This data is work in progress and can only be used for research and/or teaching purposes. It is being distributed by the RiVaL Group, University of Malta in its current state on condition that its users explicitly acknowledge the source, such as by citing our paper (bibtex in the link above), and do not distribute the data further without the express permission of the RiVaL Group.