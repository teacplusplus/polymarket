use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(XFeatures, attributes(xfeature))]
pub fn derive_xfeatures(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = input.ident;
    let generics = input.generics;

    // Ключевой фикс — поддержка generics
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let fields = match input.data {
        syn::Data::Struct(s) => s.fields,
        _ => panic!("XFeatures only works for structs"),
    };

    let mut push_calls = Vec::new();
    let mut push_n_calls = Vec::new();
    let mut name_matches = Vec::new();
    let mut field_lens = Vec::new();
    let mut field_len_n_calls = Vec::new();

    for field in fields.iter() {
        let has_attr = field.attrs.iter().any(|a| a.path().is_ident("xfeature"));
        if !has_attr {
            continue;
        }

        let ident = field.ident.as_ref().unwrap();
        let ident_str = ident.to_string();
        let ty = &field.ty;

        let len_expr = quote! { <#ty as FeatureLen>::LEN };
        field_lens.push(len_expr.clone());

        field_len_n_calls.push(quote! { <#ty as FeatureLen>::len_n(max_lag) });

        push_calls.push(quote! {
            xframe_features::push_feature(&self.#ident, &mut out);
        });

        push_n_calls.push(quote! {
            xframe_features::push_feature_n(&self.#ident, &mut out, max_lag);
        });

        name_matches.push(quote! {
            {
                let len = <#ty as FeatureLen>::LEN;
                if idx >= offset && idx < offset + len {
                    return Some(Box::leak(format!(
                        "{}{}",
                        #ident_str,
                        if len > 1 {
                            format!("[{}]", idx - offset)
                        } else {
                            String::new()
                        }
                    ).into_boxed_str()));
                }
                offset += len;
            }
        });
    }

    let expanded = quote! {
        impl #impl_generics #name #ty_generics #where_clause {
            pub fn to_x_train(&self) -> Vec<f32> {
                let mut out = Vec::new();
                #(#push_calls)*
                out
            }

            /// Как [`to_x_train`], но для лаговых массивов берёт только первые `max_lag` элементов.
            pub fn to_x_train_n(&self, max_lag: usize) -> Vec<f32> {
                let mut out = Vec::new();
                #(#push_n_calls)*
                out
            }

            pub fn feature_name(idx: usize) -> Option<&'static str> {
                let mut offset = 0usize;
                #(#name_matches)*
                None
            }

            pub fn count_features() -> usize {
                0usize #( + #field_lens )*
            }

            /// Число фичей при ограничении лаговых массивов до `max_lag`.
            pub fn count_features_n(max_lag: usize) -> usize {
                0usize #( + #field_len_n_calls )*
            }
        }
    };

    expanded.into()
}