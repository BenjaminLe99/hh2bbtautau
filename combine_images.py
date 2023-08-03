def combine_images(path_to_images, axis, dst):
    def vertical(images):
        images = [Image.open(image) for image in path_to_images]
        widths, heights = zip(*(image.size for image in images))

        max_width = max(widths)
        total_height = sum(heights)
        combined_image = Image.new("RGB", (max_width, total_height))

        offset=0
        for image in images:
            combined_image.paste(image,(0,offset))
            offset += image.size[1]
        return combined_image

    def horizontal(images):
        images = [Image.open(image) for image in path_to_images]
        widths, heights = zip(*(image.size for image in images))

        total_width = sum(widths)
        max_height = max(heights)

        combined_image = Image.new("RGB", (total_width, max_height))
        offset=0
        for image in images:
            combined_image.paste(image,(offset, 0))
            offset += image.size[0]
        return combined_image

    new_image = None
    if axis == "h":
        new_image = horizontal(path_to_images)
    elif axis == "v":
        new_image = vertical(path_to_images)
    new_image.save(dst)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="""
                    Tool to vertically or horizonally combine picture png's
                    """,
                                    prog="Prog")

    parser.add_argument("--src", "-s",
                        help="""Path to images you want to combine""",
                        type=str,
                        required=True,
                        dest="path_to_images",
                        nargs='+')

    parser.add_argument("--dst", "-d",
                        help="""Path to combined image""",
                        type=str,
                        required=True,
                        dest="dst")

    parser.add_argument("--axis", "-a",
                        help="""
                        Combine along vertical (v) or horizonal (h) axis""",
                        type=str,
                        default="v",
                        required=False,
                        dest="axis")

    args = parser.parse_args()
    combine_images(**args.__dict__)