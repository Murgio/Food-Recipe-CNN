$(document).ready(function () {
    // Init
    $('.grid').hide();
    $('.loader').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function(e) {
                $('.image-upload-wrap').hide();
                $('.file-upload-image').attr('src', e.target.result);
                $('.file-upload-content').show();
                $('.image-title').html(input.files[0].name);
            };
            reader.readAsDataURL(input.files[0]);
            // Show loading animation
            //$(this).hide();
            $('.loader').show();
            $('.grid').fadeIn(50);
            var formData = new FormData();
            formData.append('file', input.files[0]);
            // Make prediction by calling api /predict
            $.ajax({
                type: 'POST',
                url: '/predict',
                data: formData,
                contentType: false,
                cache: false,
                processData: false,
                async: true,
                success: function (data) {
                    // Get and display the result
                    $('.loader').hide();
                    $('.grid').append(data);
                    console.log('Success!');
                    //let images = document.querySelectorAll("img");
                    //lazyload(images);
                },
            });
        }
    }

    function removeUpload() {
        $('.file-upload-input').replaceWith($('.file-upload-input').clone());
        $('.file-upload-content').hide();
        $('.image-upload-wrap').show();
    }

    $('.image-upload-wrap').bind('dragover', function () {
        $('.image-upload-wrap').addClass('image-dropping');
    });
    $('.image-upload-wrap').bind('dragleave', function () {
        $('.image-upload-wrap').removeClass('image-dropping');
    });

    $(".file-upload-input").change(function () {
        $('.grid').text('');
        $('.grid').hide();
        readURL(this);
    });
});