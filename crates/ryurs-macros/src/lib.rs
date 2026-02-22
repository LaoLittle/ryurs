use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::spanned::Spanned;
use syn::{FnArg, ItemFn, LitInt, PatType, Type, parse_macro_input};

#[proc_macro_attribute]
pub fn svc(attr: TokenStream, item: TokenStream) -> TokenStream {
    let number_lit = parse_macro_input!(attr as LitInt);
    if let Err(err) = number_lit.base10_parse::<u16>() {
        return syn::Error::new(number_lit.span(), format!("invalid svc number: {err}"))
            .to_compile_error()
            .into();
    }

    let mut function = parse_macro_input!(item as ItemFn);
    let function_name = function.sig.ident.clone();
    let handler_name = format_ident!("_{}_svc_handler", function_name);
    let service_factory_name = format_ident!("{}_service", function_name);

    let mut setup = Vec::new();
    let mut call_args = Vec::new();

    for input in &mut function.sig.inputs {
        let typed = match input {
            FnArg::Typed(typed) => typed,
            FnArg::Receiver(receiver) => {
                return syn::Error::new(receiver.span(), "svc handlers must be free functions")
                    .to_compile_error()
                    .into();
            }
        };

        let x_index = match take_x_index(typed) {
            Ok(index) => index,
            Err(err) => return err.to_compile_error().into(),
        };

        if let Some(index) = x_index {
            let pat = &typed.pat;
            let ty = &typed.ty;
            setup.push(quote! {
                let #pat: #ty = context.state.read_x(#index) as #ty;
            });
            call_args.push(quote! { #pat });
            continue;
        }

        if is_mut_svc_context(&typed.ty) {
            call_args.push(quote! { context });
            continue;
        }

        return syn::Error::new(
            typed.span(),
            "svc arguments must be #[x(N)] or &mut SvcContext",
        )
        .to_compile_error()
        .into();
    }

    let expanded = quote! {
        #function

        const fn #service_factory_name() -> SvcService {
            fn #handler_name(context: &mut SvcContext<'_>) -> SvcResult {
                #(#setup)*
                #function_name(#(#call_args),*)
            }

            SvcService {
                svc_number: #number_lit,
                name: stringify!(#function_name),
                handler: #handler_name,
            }
        }
    };

    expanded.into()
}

fn take_x_index(parameter: &mut PatType) -> syn::Result<Option<u8>> {
    let mut x_index = None;
    let mut retained = Vec::with_capacity(parameter.attrs.len());

    for attribute in parameter.attrs.drain(..) {
        if attribute.path().is_ident("x") {
            if x_index.is_some() {
                return Err(syn::Error::new(
                    attribute.span(),
                    "duplicate #[x(..)] attribute",
                ));
            }
            let lit = attribute.parse_args::<LitInt>()?;
            x_index = Some(lit.base10_parse::<u8>()?);
        } else {
            retained.push(attribute);
        }
    }

    parameter.attrs = retained;
    Ok(x_index)
}

fn is_mut_svc_context(ty: &Type) -> bool {
    let Type::Reference(reference) = ty else {
        return false;
    };
    if reference.mutability.is_none() {
        return false;
    }
    let Type::Path(path) = &*reference.elem else {
        return false;
    };

    path.path
        .segments
        .last()
        .map(|segment| segment.ident == "SvcContext")
        .unwrap_or(false)
}
