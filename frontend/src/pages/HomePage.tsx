import { useState } from "react";
import api from "../api/axios";


function Home() {
    const [file, setFile] = useState<File | null>(null)

    const uploadFile = async () => {
        if (!file) return;
        const form = new FormData();
        form.append("file", file);
        try {
            const res = await api.post('/detect-face', form, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            console.log(res?.data);
        } catch (error) {
            console.error('Upload failed:', error);
        }
    }

    return (
        <div>
            <input type="file" onChange={(e) => setFile(e.target.files?.[0] || null)} />
            <button onClick={uploadFile} disabled={!file}>Send</button>
        </div>
    )
}
export default Home