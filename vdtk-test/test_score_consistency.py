def test_coco_score_consistency():
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
    import os
    from vdtk.score import _ciderd, _bleu
    import numpy as np

    # Get the pycocoevalcap scores

    annotation_file = os.path.join(os.path.dirname(__file__), "test_assets", "captions_val2014.json")
    results_file = os.path.join(os.path.dirname(__file__), "test_assets", "captions_val2014_fakecap_results.json")
    dataset_file = os.path.join(os.path.dirname(__file__), "test_assets", "dataset.json")

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    coco_eval.params["image_id"] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # Get the vdtk scores
    vdtk_cider_scores = _ciderd([dataset_file], None)
    np.testing.assert_almost_equal(coco_eval.eval["CIDEr"], vdtk_cider_scores[0][0], decimal=3)

    vdtk_bleu_scores = _bleu([dataset_file], None)
    np.testing.assert_almost_equal(coco_eval.eval["Bleu_1"], vdtk_bleu_scores[0][0][0], decimal=3)
    np.testing.assert_almost_equal(coco_eval.eval["Bleu_2"], vdtk_bleu_scores[0][0][1], decimal=3)
    np.testing.assert_almost_equal(coco_eval.eval["Bleu_3"], vdtk_bleu_scores[0][0][2], decimal=3)
    np.testing.assert_almost_equal(coco_eval.eval["Bleu_4"], vdtk_bleu_scores[0][0][3], decimal=3)


if __name__ == "__main__":
    test_coco_score_consistency()
