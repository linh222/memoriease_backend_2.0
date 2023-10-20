def extract_query_blip_embedding(query, model, processor):
    text_input = processor['eval'](query)
    data = {'text_input': text_input}
    text_features = model.extract_features(data, mode='text')
    return text_features.text_embeds_proj[:, 0, :].squeeze().cpu().numpy()

# if __name__ == "__main__":
#     from LAVIS.lavis.models import load_model_and_preprocess
#
#     query = 'I am at school'
#     device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
#     model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor",
#                                                                       model_type="coco", is_eval=True,
#                                                                       device=device)
#     text_input = txt_processors["eval"](query)
#     sample = {"text_input": [text_input]}
#     result = extract_query_blip_embedding(query, model, txt_processors)
#     print(result.shape)
